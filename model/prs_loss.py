import torch.nn as nn
import torch

def hamilton_product(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    output = torch.stack((w, x, y, z), dim=-1)
    return output

def quat_conjugate(q):
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj

def quat_rotate_vector(q, v):
    # 将向量转换为四元数，实部为 0
    q_v = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    
    # 四元数乘法进行旋转： q * v * q^(-1)
    q_conj = quat_conjugate(q)
    v_rotated = hamilton_product(hamilton_product(q, q_v), q_conj)
    return v_rotated[..., 1:]  # 返回旋转后的向量部分

def sym_plane_tran(points,planes):
    batch_num = planes.shape[0]
    planes_num = planes.shape[1]
    points_num = points.shape[1]
    ns = planes[..., 0:-1]
    ds = planes[..., -1]
    ns_norm = torch.norm(ns, p=2, dim=2, keepdim=True)
    ds = ds/torch.norm(ns, p=2, dim=2)
    ns = ns/ns_norm
    ns = ns.unsqueeze(2).repeat(1,1,points_num,1)
    ds = ds.unsqueeze(2).repeat(1,1,1,points_num).reshape(-1,planes_num,points_num,1)
    points = points.unsqueeze(2).repeat(1,1,planes_num,1,1).view(batch_num,planes_num,points_num,3)
    return points - 2*(torch.sum(points*ns,dim=3,keepdim=True)+ds)*ns

def sym_quad_tran(points,quads):
    quads_num = quads.shape[1]
    points_num = points.shape[1]
    batch_num = quads.shape[0]
    qs = quads[..., 1:]
    qs_norm = torch.norm(qs, p=2, dim=2, keepdim=True)
    qs = qs/qs_norm
    qs = torch.cat([torch.ones((batch_num,quads_num,1)).float().to(qs.device),qs],dim=-1)
    qs = 0.707*qs
    qs = qs.unsqueeze(2).repeat(1,1,points_num,1)
    mid_point = torch.mean(points, dim=1)
    points = points.unsqueeze(2).repeat(1,1,quads_num,1,1).view(batch_num,quads_num,points_num,3)
    points = points - mid_point.unsqueeze(1).repeat(1,1,quads_num,points_num).view(batch_num,quads_num,points_num,3)
    # # 将向量转换为四元数，实部为 0
    # q_v = torch.cat((torch.zeros_like(points[..., :1]), points), dim=-1)
    # q_v,qs
    # # 四元数乘法进行旋转： q * v * q^(-1)
    # q_conj = quat_conjugate(qs)
    # v_rotated = hamilton_product(hamilton_product(qs, q_v), q_conj)
    # points = v_rotated[..., 1:]  # 返回旋转后的向量部分
    points = quat_rotate_vector(qs, points)
    return points

class SymQuadLoss(nn.Module):
    def __init__(self):
        super(SymQuadLoss, self).__init__()
    
    def forward(self, voxel, points, closest_points, quads):
        quads_num = quads.shape[1]
        batch_num = quads.shape[0]
        points = sym_quad_tran(points, quads)
        grid_size = voxel.shape[1]
        indexs = torch.ceil((points+0.5)*grid_size-0.5).long()
        indexs = indexs[...,0]*grid_size**2+indexs[...,1]*grid_size+indexs[...,2]
        indexs = torch.clamp(indexs,0,grid_size**3-1)
        cp = torch.gather(closest_points.unsqueeze(1).repeat(1,quads_num,1,1),
                                2,indexs.unsqueeze(3).repeat(1,1,1,3))
        mask = 1 - torch.gather(voxel.view(batch_num,1,-1).repeat(1,quads_num,1),
                                        2,indexs)
        distance = (points-cp)*mask.unsqueeze(3).repeat(1,1,1,3)
        return torch.mean(torch.sum(distance**2,dim=3))

class SymPlaneLoss(nn.Module):
    def __init__(self):
        super(SymPlaneLoss, self).__init__()
    
    # batch size
    def forward(self, voxel, points, closest_points, planes):
        batch_num = planes.shape[0]
        planes_num = planes.shape[1]
        points = sym_plane_tran(points,planes)
        grid_size = voxel.shape[1]
        indexs = torch.ceil((points+0.5)*grid_size-0.5).long()
        indexs = indexs[...,0]*grid_size**2+indexs[...,1]*grid_size+indexs[...,2]
        indexs = torch.clamp(indexs,0,grid_size**3-1)
        cp = torch.gather(closest_points.unsqueeze(1).repeat(1,planes_num,1,1),
                                2,indexs.unsqueeze(3).repeat(1,1,1,3))
        mask = 1 - torch.gather(voxel.view(batch_num,1,-1).repeat(1,planes_num,1),
                                        2,indexs)
        distance = (points-cp)*mask.unsqueeze(3).repeat(1,1,1,3)
        return torch.mean(torch.sum(torch.sum(distance**2,dim=3),dim=2))
    
class ReLoss(nn.Module):
    def __init__(self):
        super(ReLoss, self).__init__()
    
    def forward(self, vectors):
        batch_num = vectors.shape[0]
        l2_norm = torch.norm(vectors, p=2, dim=2, keepdim=True)
        vectors = vectors/l2_norm
        mat = torch.matmul(vectors.transpose(1, 2), vectors)

        loss = (mat - torch.eye(mat.shape[1]).repeat(batch_num,1,1).to(mat.device))**2
        return torch.mean(torch.sum(loss))

class PrsLoss(nn.Module):
    def __init__(self):
        super(PrsLoss, self).__init__()
        self.sym_plane_loss = SymPlaneLoss()
        self.sym_quad_loss = SymQuadLoss()
        self.re_loss = ReLoss()
    
    def forward(self, voxel, points, closest_points, planes, quads):
        batch_num = voxel.shape[0]
        closest_points = closest_points.view(batch_num,-1,3)
        # print(closest_points.shape)
        # self.sym_quad_loss(voxel, points, closest_points, quads) \
        #      + self.re_loss(quads[:,1:])*0.2 \
        return  50*self.sym_plane_loss(voxel, points, closest_points, planes) \
             + self.re_loss(planes[:,:-1]) \
             + 50*self.sym_quad_loss(voxel, points, closest_points, quads) \
             + self.re_loss(quads[:,1:]) \
    
if __name__ == '__main__':
    loss = PrsLoss()
    voxel = torch.randint(0,2,(32,32,32,32))
    points = torch.rand(32,100,3)-0.5
    planes = torch.rand(32,3,4)
    closest_points = torch.rand(32,32,32,32,3)-0.5
    print(loss(voxel, points, closest_points, planes, planes))