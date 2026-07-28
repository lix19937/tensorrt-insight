
import torch

voxel_features = torch.randn(1, 64, 1, 40000)  # (B, C, H, W)   
voxel_features_bk = voxel_features.clone()

# original
vf1 = voxel_features.permute(0, 2, 3, 1).contiguous()
P, C = vf1.size(-2), vf1.size(-1)
vf1 = vf1.reshape(P, C)
voxels1 = vf1.t()

# after exp
C = voxel_features.size(1) 
voxels2 = voxel_features.transpose(0, 1).reshape(C, -1)

voxels3 = voxel_features_bk.permute(1, 0, 2, 3).reshape(C, -1)

voxels4 = voxel_features_bk.view(1, 64, 40000).permute(1, 0, 2).reshape(C, -1)

voxels5 = voxel_features_bk.reshape(C, -1)


# verify
print(torch.allclose(voxels1, voxels2))  # True

print(torch.equal(voxels1, voxels2))     # True

print(torch.equal(voxels1, voxels3))     # True

print(torch.equal(voxels1, voxels4))     # True

print(torch.equal(voxels1, voxels5))     # True
