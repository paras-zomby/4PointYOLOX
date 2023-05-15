import torch
import torch.nn as nn


class IOULoss(nn.Module):
    def __init__(self, reduction="none"):
        super(IOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        raise Exception("IOU Loss can not be used!")

    @staticmethod
    def __is_inside(p1, p2, q):
        r = (p2[..., 0] - p1[..., 0]) * (q[..., 1] - p1[..., 1]) - (
                p2[..., 1] - p1[..., 1]) * (
                    q[..., 0] - p1[..., 0])
        return r <= 0

    @staticmethod
    def __compute_intersection(p1, p2, p3, p4):
        intersection = torch.empty_like(p1)

        # !calc m1, b1, m2, b2
        m1 = (p2[..., 1] - p1[..., 1]) / (p2[..., 0] - p1[..., 0])
        b1 = p1[..., 1] - m1 * p1[..., 0]
        # slope and intercept of second line
        m2 = (p4[..., 1] - p3[..., 1]) / (p4[..., 0] - p3[..., 0])
        b2 = p3[..., 1] - m2 * p3[..., 0]

        # !make condition tensors
        mask1 = p2[..., 0] == p1[..., 0]
        mask2 = p4[..., 0] == p3[..., 0]
        part1 = mask1  # !if p2[..., 0] == p1[..., 0]:
        part2 = ~mask1 & mask2  # !elif p4[..., 0] == p3[..., 0]:
        part3 = ~mask1 & ~mask2  # !else:

        # !part1
        temp = intersection[part1]
        temp[:, 0] = p1[part1][:, 0]
        # y-coordinate of intersection
        temp[:, 1] = m2[part1] * p1[part1][:, 0] + b2[part1]
        intersection[part1] = temp

        # !part2
        temp = intersection[part2]
        temp[:, 0] = p3[part2][:, 0]
        # y-coordinate of intersection
        temp[:, 1] = m1[part2] * p3[part2][:, 0] + b1[part2]
        intersection[part2] = temp

        # !part3
        temp = intersection[part3]
        # x-coordinate of intersection
        temp[:, 0] = (b2[part3] - b1[part3]) / (m1[part3] - m2[part3])
        # y-coordinate of intersection
        temp[:, 1] = m1[part3] * (b2[part3] - b1[part3]) / (m1[part3] - m2[part3]) + b1[part3]
        intersection[part3] = temp
        # print(intersection.requires_grad)
        # print(temp.requires_grad)

        # need to unsqueeze so torch.cat doesn't complain outside func
        return intersection

    @staticmethod
    def _clip(subject_polygon, clipping_polygon):
        # it is assumed that requires_grad = True only for clipping_polygon
        # subject_polygon and clipping_polygon are ... x N x 2 and ... x M x 2 torch
        # tensors respectively

        # !make all tensors on the same device
        device = subject_polygon.device
        final_polygon = torch.clone(subject_polygon)
        point_num = torch.empty(
                *final_polygon.shape[:-2], dtype=torch.long, device=device
                ).fill_(4)
        for i in range(4):  # *没用clipping_polygon.shape[-2])，因为确定会有四个结果

            # stores the vertices of the next iteration of the clipping procedure
            # final_polygon consists of list of 1 x 2 tensors
            next_polygon = torch.clone(final_polygon)
            next_point_num = torch.clone(point_num)
            # stores the vertices of the final clipped polygon. This will be
            # a K x 2 tensor, so need to initialize shape to match this
            final_polygon = torch.empty(*next_polygon.shape[:-2], 10, 2, device=device)
            point_num = torch.zeros(
                    *next_polygon.shape[:-2], device=device, dtype=torch.long
                    )

            # these two vertices define a line segment (edge) in the clipping
            # polygon. It is assumed that indices wrap around, such that if
            # i = 0, then i - 1 = M.
            # c_edge_start = clipping_polygon[..., i - 1, :]
            # c_edge_end = clipping_polygon[..., i, :]

            for j in range(
                    next_point_num.max() if next_point_num.numel() > 0 else 0
                    ):  # *两个四边形最多有8个交点，所以没用next_polygon.shape[-2]。
                # these two vertices define a line segment (edge) in the subject
                # polygon
                mask = next_point_num > j
                index = (next_point_num[mask] + j - 1) % next_point_num[
                    next_point_num > j]
                s_edge_start = next_polygon[mask, index]
                s_edge_end = next_polygon[mask][:, j, :]
                c_edge_start = clipping_polygon[mask][:, i - 1, :]
                c_edge_end = clipping_polygon[mask][:, i, :]

                final_polygon_this_circle = final_polygon[mask]
                point_num_this_circle = point_num[mask]

                condition_1 = IOULoss.__is_inside(c_edge_start, c_edge_end, s_edge_end)
                condition_2 = IOULoss.__is_inside(c_edge_start, c_edge_end, s_edge_start)
                part1 = condition_1 & ~condition_2
                part2 = condition_1
                part3 = ~condition_1 & condition_2

                intersection_part1 = IOULoss.__compute_intersection(
                        s_edge_start[part1], s_edge_end[part1], c_edge_start[part1],
                        c_edge_end[part1]
                        )
                final_polygon_this_circle[
                    part1, point_num_this_circle[part1]] = intersection_part1.float()
                point_num_this_circle[part1] += 1

                final_polygon_this_circle[part2, point_num_this_circle[part2]] = \
                    s_edge_end[part2]
                point_num_this_circle[part2] += 1

                intersection_part3 = IOULoss.__compute_intersection(
                        s_edge_start[part3], s_edge_end[part3], c_edge_start[part3],
                        c_edge_end[part3]
                        )
                final_polygon_this_circle[
                    part3, point_num_this_circle[part3]] = intersection_part3
                point_num_this_circle[part3] += 1

                final_polygon[mask] = final_polygon_this_circle
                point_num[mask] = point_num_this_circle
                # print(f"i = {i}, j = {j}", final_polygon)
        return final_polygon, point_num

    @staticmethod
    def _calc_area(points, point_nums):
        area = torch.zeros_like(point_nums, dtype=torch.float)
        for i in range(8):
            mask = point_nums > i
            index = (point_nums[mask] + i - 1) % point_nums[point_nums > i]
            area[mask] += torch.stack(
                    (points[mask][:, i],
                     points[mask, index]), dim=-2
                    ).det()
        return area * 0.5

    @staticmethod
    def _calc_area_fixnum(points, num=4):
        area = torch.zeros(*points.shape[:-2], dtype=torch.float, device=points.device)
        for i in range(num):
            area += torch.stack(
                    (points[..., i, :],
                     points[..., i - 1, :]), dim=-2
                    ).det()
        return area * 0.5

    @staticmethod
    def _get_bounding_box(points):
        box_l = torch.min(points[..., 0], dim=-1).values
        box_r = torch.max(points[..., 0], dim=-1).values
        box_t = torch.min(points[..., 1], dim=-1).values
        box_b = torch.max(points[..., 1], dim=-1).values
        box_t_l = torch.stack((box_l, box_t), dim=-1)
        box_b_l = torch.stack((box_l, box_b), dim=-1)
        box_b_r = torch.stack((box_r, box_b), dim=-1)
        box_t_r = torch.stack((box_r, box_t), dim=-1)
        return torch.stack((box_t_l, box_b_l, box_b_r, box_t_r), dim=-2)

    @staticmethod
    def _if_concave_quadrangle(points):
        res = torch.ones(*points.shape[:-2], dtype=torch.float, device=points.device)
        for i in range(4):
            res *= torch.stack(
                    (points[..., i - 1, :] - points[..., i, :],
                     points[..., (i + 1) % 4, :] - points[..., i, :]
                     ), dim=-2
                    ).det()
        return res < 0

    @staticmethod
    def _if_points_right_order(points):
        a = points[..., 0, :]
        b = points[..., 1, :]
        c = points[..., 2, :]
        d = points[..., 3, :]
        v1 = torch.stack((a - b, d - b), dim=-2).det()
        v2 = torch.stack((d - b, c - b), dim=-2).det()
        v3 = torch.stack((b - c, a - c), dim=-2).det()
        v4 = torch.stack((a - c, d - c), dim=-2).det()
        # !cancave rectangles are always in right order.
        return (v1 * v2 > 0) & (v3 * v4 > 0)

    @staticmethod
    def __if_in_poly(poly, point):
        num = poly.shape[-2]
        res = torch.zeros(*poly.shape[:-2], num, device=poly.device)
        for i in range(num):
            res[..., i] = torch.stack(
                    (point - poly[..., i, :],
                     poly[..., (i + 1) % num, :] - poly[..., i, :]), dim=-2
                    ).det()
        return ((res > 0).sum(dim=-1) == num) | ((res < 0).sum(dim=-1) == num)

    @staticmethod
    def if_inside_bounding_box(point, boxes):
        bounding_box = IOULoss._get_bounding_box(boxes.view(*boxes.shape[:-1], 4, 2))
        return IOULoss.__if_in_poly(bounding_box, point)

    @staticmethod
    def get_IOU(pred_box, gt_box, train_mode=True):
        """only pred_box needs gard"""
        # !change the coordinates format from [8] to [4, 2]
        pboxes = torch.stack(pred_box.split(2, dim=-1), dim=-2)
        gboxes = torch.stack(gt_box.split(2, dim=-1), dim=-2)
        if train_mode:
            mask = IOULoss._if_concave_quadrangle(pboxes)
            pboxes[mask] = IOULoss._get_bounding_box(pboxes[mask])
            mask = IOULoss._if_points_right_order(pboxes)
            # ! if not in right order, then exchange the 'c', 'd'
            pboxes[~mask] = pboxes[~mask][:, [0, 1, 3, 2]]
            # ! if still not in right order
            mask = IOULoss._if_points_right_order(pboxes)
            # ! then continue exchange the 'b', 'd'
            pboxes[~mask] = pboxes[~mask][:, [0, 2, 1, 3]]
            clipped_res, res_points_num = IOULoss._clip(gboxes, pboxes)
            area_i = torch.abs(IOULoss._calc_area(clipped_res, res_points_num))
            return area_i / (torch.abs(IOULoss._calc_area_fixnum(pboxes)) +
                             IOULoss._calc_area_fixnum(gboxes) - area_i + 1e-16)
        else:
            mask1 = IOULoss._if_concave_quadrangle(pboxes)
            mask2 = IOULoss._if_points_right_order(pboxes)
            mask3 = IOULoss._calc_area_fixnum(pboxes) < 0
            clipped_res, res_points_num = IOULoss._clip(gboxes, pboxes)
            area_i = torch.abs(IOULoss._calc_area(clipped_res, res_points_num))

            res_iou = area_i / (IOULoss._calc_area_fixnum(pboxes) +
                                IOULoss._calc_area_fixnum(gboxes) - area_i + 1e-16)
            res_iou[mask1 | ~mask2 | mask3] = 0
            return res_iou


if __name__ == "__main__":
    iou = IOULoss()
    # squares
    # subject_polygon = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    # clipping_polygon = [(0, 0), (0, 2), (2, 2), (2, 0)]

    # squares: different order of points
    # subject_polygon = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    # clipping_polygon = [(2, 0), (0, 0), (0, 2), (2, 2)]

    # subject_polygon = torch.tensor(subject_polygon).float()
    # clipping_polygon = torch.tensor(clipping_polygon).float()
    # clipped_polygon, points_num = iou._clip(
    #         subject_polygon[None, None, ...],
    #         clipping_polygon[None, None, ...]
    #         )
    #
    # for i in range(9):
    #     if clipped_polygon[points_num == i][:, :i].shape[0] != 0:
    #         print(f"clipped polygon that points num = {i}s:")
    #         print(clipped_polygon[points_num == i][:, :i])
    # print("areas = ", iou._calc_area(clipped_polygon, points_num), sep='\n')
    #
    # points = torch.Tensor(
    #         [[171.9181, 373.2570, 171.0784, 384.3878, 189.7984, 387.2672, 190.3968,
    #         375.9795]]
    #         )
    # pointx = torch.randn(1, 10, 1) * 10 + 175
    # pointy = torch.randn(1, 10, 1) * 5 + 375
    # print(
    #         iou.if_inside_box(
    #                 torch.stack((pointx, pointy), dim=-1), points[None, ...].expand(1, 10, 8)
    #                 )
    #         )

    # ploy_1 = torch.tensor([19, 155, 475, 139, 247, 145, 474, 138]).float().unsqueeze(
    #         0
    #         ).unsqueeze(0)
    # ploy_2 = torch.tensor([407, 47, 19, 254, 462, 7, 394, 255]).float().unsqueeze(
    #         0
    #         ).unsqueeze(0)
    #
    # ploy_2 = torch.stack(ploy_2.split(2, dim=-1), dim=-2)
    # mask = IOULoss._if_points_right_order(ploy_2)
    # ploy_2[~mask] = ploy_2[~mask][:, [0, 1, 3, 2]]
    # # ! if still not in right order
    # mask = IOULoss._if_points_right_order(ploy_2)
    # # ! then continue exchange the 'b', 'd'
    # ploy_2[~mask] = ploy_2[~mask][:, [0, 2, 1, 3]]
    # if IOULoss._if_concave_quadrangle(ploy_2):
    #     ploy_2 = IOULoss._get_bounding_box(ploy_2)
    # if IOULoss._calc_area_fixnum(ploy_2).item() < 0:
    #     ploy_2 = ploy_2[..., [3, 2, 1, 0], :]
    #
    # print(iou.get_IOU(ploy_1, ploy_2.view(1, 1, 8)))

    for i in range(int(1000)):
        ploy_1 = torch.randint(0, 480, (100, 100, 8)).float()
        ploy_2 = torch.randint(0, 480, (100, 100, 8)).float()

        ploy_2 = torch.stack(ploy_2.split(2, dim=-1), dim=-2)
        mask = IOULoss._if_concave_quadrangle(ploy_2)
        ploy_2[mask] = IOULoss._get_bounding_box(ploy_2[mask])
        mask = IOULoss._if_points_right_order(ploy_2)
        ploy_2[~mask] = ploy_2[~mask][:, [0, 1, 3, 2]]
        # ! if still not in right order
        mask = IOULoss._if_points_right_order(ploy_2)
        # ! then continue exchange the 'b', 'd'
        ploy_2[~mask] = ploy_2[~mask][:, [0, 2, 1, 3]]
        mask = IOULoss._calc_area_fixnum(ploy_2) < 0
        ploy_2[mask] = ploy_2[mask][:, [3, 2, 1, 0]]
        ploy_2 = ploy_2.view(*ploy_2.shape[:-2], 8)

        if i % 100 == 0:
            print(f'now i = {i}')

        if (iou.get_IOU(ploy_1, ploy_2, True) < 0).sum() > 0:
            raise Exception
