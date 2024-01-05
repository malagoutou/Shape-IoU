def shape_iou(box1, box2, xywh=True, scale=0, eps=1e-7):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2

    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    
    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU
    iou = iou - distance - 0.5 * ( shape_cost)
    return iou  # IoU
