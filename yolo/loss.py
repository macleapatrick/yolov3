import torch
import torch.nn as nn
from classweights import WEIGHTS

class YOLOv3Loss(nn.Module):
    def __init__(self, anchors, batch_size=1, num_classes=90, resolution=416, grid=[13, 26, 52]):
        super(YOLOv3Loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Mean squared error loss for bounding box regression
        self.bce_loss = nn.BCELoss(reduction='none')  # Binary cross entropy loss for objectness and class predictions
        self.sigmoid = nn.Sigmoid()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.anchors = torch.tensor(anchors, dtype=torch.int32, device=self.device)
        self.num_anchors = torch.tensor(len(anchors), dtype=torch.int32, device=self.device)
        self.num_classes = torch.tensor(num_classes, dtype=torch.int32, device=self.device)
        self.resolution = torch.tensor(resolution, dtype=torch.int32, device=self.device)
        self.grids = torch.tensor(grid, dtype=torch.int32, device=self.device)
        self.grid_res = torch.tensor([resolution // grid[i] for i in range(len(grid))], device=self.device)
        self.scales = torch.tensor(len(grid), dtype=torch.int32, device=self.device)
        self.batch_size = torch.tensor(batch_size, dtype=torch.int32, device=self.device)
        self.lamd_noobj = torch.tensor(0.25, device=self.device)
        self.lamd_obj = torch.tensor(2, device=self.device)
        self.lamd_coord = torch.tensor(0.5, device=self.device)
        self.lamb_class = torch.tensor(2.0, device=self.device)

        self._init_bases()
        self._init_classweights()

    def forward(self, predictions, targets):
        # predictions: (batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
        # targets: (batch_size, targets, info) where info is (obj, x, y, w, h, scale_num, anchor_num, x_grid, y_grid)

        batch_size, _, _ = targets.size()
        _, _, grid_size, _, _ = predictions.size()

        scale = torch.nonzero(self.grids == grid_size)[0]

        # define loss tensor
        loss = torch.zeros((batch_size), device=self.device)

        # transform targets to match predictions
        targets, t_cls = self.transform_targets(targets, scale)

        t_obj, t_xc, t_yc, t_w, t_h = targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3], targets[..., 4]

        # form object masks
        has_obj_mask = t_obj != 0
        no_obj_mask = t_obj == 0

        # transform predictions to absolute values
        predictions = self.transform_predictions(predictions)

        p_xc, p_yc, p_w, p_h, p_cf, p_cls = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3], predictions[..., 4], predictions[..., 5:]

        # bounding box loss
        bb_loss = self.lamd_coord * (self.mse_loss(has_obj_mask * p_xc, t_xc) + 
                                     self.mse_loss(has_obj_mask * p_yc, t_yc) + 
                                     self.mse_loss(has_obj_mask * p_w, t_w) + 
                                     self.mse_loss(has_obj_mask * p_h, t_h))

        # objectness loss
        has_object_loss = self.lamd_obj * self.bce_loss(has_obj_mask * p_cf, has_obj_mask.float())
        no_object_loss = self.lamd_noobj * self.bce_loss(no_obj_mask * p_cf, has_obj_mask.float())

        # class loss - scale by freq of classes in dataset
        class_loss =  has_obj_mask * self.lamb_class * torch.sum(self.clsw[scale] * self.bce_loss(p_cls, t_cls), dim=4)

        # total loss
        loss = torch.mean(bb_loss + has_object_loss + no_object_loss + class_loss)

        # scale component losses - these are reported just for monitoring reasons
        bb_loss = torch.sum(bb_loss) / has_obj_mask.sum().clamp(min=1)
        has_object_loss = torch.sum(has_object_loss) / has_obj_mask.sum().clamp(min=1)
        no_object_loss = torch.sum(no_object_loss) / no_obj_mask.sum().clamp(min=1)
        class_loss = torch.sum(class_loss) / has_obj_mask.sum().clamp(min=1)

        return loss, bb_loss, has_object_loss, no_object_loss, class_loss
    
    def transform_targets(self, in_targets, scale):
        # transforms targets with defined anchors to match dimensionalty of predictions for easy loss calculation as
        # well as returns the one hot encoded class labels in the same format
        # in_targets - (batch_size, targets, info) where info is (obj, x, y, w, h, scale_num, anchor_num, x_grid, y_grid)
        # scale - the scale of the grid size of the predictions

        batch_size, target_size, info_size = in_targets.size()
        grid_size = self.grids[scale]

        # define output tensor
        out_targets = torch.zeros((batch_size, self.num_anchors, grid_size, grid_size, info_size), device=self.device)
        one_hot = torch.zeros((batch_size, self.num_anchors, grid_size, grid_size, self.num_classes), device=self.device)

        # Filter relevant targets based on object presence and scale
        mask = (in_targets[..., 0] != 0) & (in_targets[..., 5] == scale)
        relevant_targets = in_targets[mask]
        relevant_targets = relevant_targets.float()

        # Extract relevant components for assignment
        batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1).expand(-1, target_size)[mask]
        batch_indices = batch_indices.long()
        anchor_nums = relevant_targets[:, 6].long()
        x_grids = relevant_targets[:, 7].long()
        y_grids = relevant_targets[:, 8].long()

        # Populate out_targets tensor
        out_targets[batch_indices, anchor_nums, y_grids, x_grids] = relevant_targets

        # Populate one_hot tensor
        obj_classes = relevant_targets[:, 0].long() - 1  # Convert class to zero-indexed
        one_hot[batch_indices, anchor_nums, y_grids, x_grids, obj_classes] = 1

        return out_targets, one_hot

    def transform_predictions(self, predictions):
        # transform relative predictions at each grid location to absolute values on the images
        # convert objectness and class predictions to probabilities
        # normally wouldn't do this in the loss function, but it is easier to do it here than in the model
        _, _, grid_size, _, _ = predictions.size()

        scale = torch.nonzero(self.grids == grid_size)[0]
        grid_res = self.grid_res[scale]

        predictions[...,  0] = grid_res * self.sigmoid(predictions[...,  0]) + grid_res * self.cx[scale]
        predictions[...,  1] = grid_res * self.sigmoid(predictions[...,  1]) + grid_res * self.cy[scale]
        predictions[...,  2] = torch.exp(predictions[...,  2]) * self.anchor_width[scale]
        predictions[...,  3] = torch.exp(predictions[...,  3]) * self.anchor_height[scale]
        predictions[...,  4] = self.sigmoid(predictions[...,  4])
        predictions[..., 5:] = self.sigmoid(predictions[..., 5:])

        return predictions

    def _init_bases(self):
        # init these values once when object is initalized, so we dont have to do it every time forward is called
        # they are used in the transformtion of the relative predictions to absolute values on the images

        self.cx = []
        self.cy = []
        for scale in range(self.scales):
            grid_size = self.grids[scale]
            base = torch.arange(grid_size).repeat(grid_size, 1)
            cx = base.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_anchors, grid_size, grid_size)
            cy = base.t().unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_anchors, grid_size, grid_size)
            cx = cx.to(self.device)
            cy = cy.to(self.device)
            self.cx.append(cx)
            self.cy.append(cy)

        self.anchor_width = []
        self.anchor_height = []
        for scale in range(self.scales):
            grid_size = self.grids[scale]
            anchor_width = torch.stack([torch.tile(torch.tensor(self.anchors[scale][i][0]), dims=(grid_size, grid_size)) for i in range(self.num_anchors)]).expand(self.batch_size, self.num_anchors, grid_size, grid_size)
            anchor_height = torch.stack([torch.tile(torch.tensor(self.anchors[scale][i][1]), dims=(grid_size, grid_size)) for i in range(self.num_anchors)]).expand(self.batch_size, self.num_anchors, grid_size, grid_size)
            anchor_width = anchor_width.to(self.device)
            anchor_height = anchor_height.to(self.device)
            self.anchor_width.append(anchor_width)
            self.anchor_height.append(anchor_height)

        return None
    
    def _init_classweights(self):
        # expand into the dimensionality of the predictions
        self.clsw = []

        weights = torch.tensor([WEIGHTS[key] for key in sorted(WEIGHTS.keys())])
        sum = torch.sum(weights)

        for scale in range(self.scales):
            grid_size = self.grids[scale]
            clsw = torch.zeros((self.batch_size, self.num_anchors, grid_size, grid_size, self.num_classes))
            clsw[:, :, :, :] = 1 / (torch.tensor(weights) / sum) / 90
            clsw[clsw == float("Inf")] = 1
            clsw = clsw.to(self.device)
            self.clsw.append(clsw)

        return None
    
        

        

                    



        


