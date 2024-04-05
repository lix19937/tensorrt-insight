## 完整的decoder 结构图     

![workflow](./workflow.png)      
![sv_block_dataflow](./sv_block_dataflow.png)      
feat1 `6, 256, h, w`      
feat2 `6, 256, h//2, w//2`    
feat3 `6, 256, h//4, w//4`     
feat4 `6, 256, h//8, w//8`   

```py
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of SVDetCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape        `(num_query, bs, embed_dims)`.
            key (Tensor): The key tensor with shape                `(num_key,   bs, embed_dims)`.
            value (Tensor): The value tensor with shape            ` !!! `. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.    Default: None.
            key_pos (Tensor): The positional encoding for `key`.        Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4), all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add additional two dimensions is (w, h) to form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in different level. With shape  (num_levels, 2), last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None: # 这里 key 为空， 在ca 中没有用到    
            key = query
        if value is None: # 这里 value 有内容， 是fpn的特征图   
            value = key

        if residual is None: # 这里 residual 为空
            inp_residual = query

        if query_pos is not None: # 这里 query_pos 有内容
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        # reference_points_3d 与 reference_points 关系  ？  其实是一样的
        # reference_points (bs, seq_len, 3)
        reference_points_3d, output, mask = feature_sampling_onnx(value, reference_points, self.pc_range, kwargs['img_shape'], kwargs['lidar2img'])

        attention_weights = attention_weights.sigmoid() * mask

        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        
        output = self.output_proj(output)

        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return output + inp_residual + pos_feat
```

```py
def feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img):
    lidar2img = lidar2img.type_as(mlvl_feats[0])
    # lidar2img = reference_points.new_tensor(lidar2img) # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # ori
    # reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    # eps = 1e-5
    # mask = (reference_points_cam[..., 2:3] > eps)
    # reference_points_cam = reference_points_cam[..., 0:2] / torch.clamp(reference_points_cam[..., 2:3], eps)
    # reference_points_cam[..., 0] /= img_shape[0][1]
    # reference_points_cam[..., 1] /= img_shape[0][0]

    # whr version
    img_shapes = lidar2img.new_tensor([img_shape[0][1], img_shape[0][0], 1, 1])[None, None, None, :].repeat(B, num_cam, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1) / img_shapes
    mask = reference_points_cam[..., 2:3] > 1e-2
    reference_points_cam = torch.clamp(
                                        torch.where(mask,
                                                reference_points_cam[..., 0:2]/torch.clamp(reference_points_cam[..., 2:3],min=0.01),
                                                mask.new_tensor(torch.ones_like(reference_points_cam[..., 0:2]))*(-1.)
                                                ),
                                            min=-1., max=2.)

    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)
        # sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = bilinear_grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
```
