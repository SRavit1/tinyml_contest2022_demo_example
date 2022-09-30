import torch

def pack(arr, pckWdt=32):
    if not len(arr)%pckWdt == 0:
        return None
    packs = []
    for i in range(int(len(arr)/pckWdt)):
        pack_dec = 0
        for j in range(pckWdt):
            pack_dec += arr[i*pckWdt + j]*pow(2, pckWdt-1-j) if arr[i*pckWdt + j] == 1 else 0
        pack = str(hex(int(pack_dec)))
        packs.append(pack)
    return ", ".join(packs)

def convert_fc_act(x, binarize=True):
    x = x[0].flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_conv_act(x, binarize=True):
    x = x[0]
    x = x.permute((1, 2, 0)) #CHW -> HWC, BWN
    x = x.flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_fc_weight(w):
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_conv_weight(w):
    w = w.permute((0, 3, 2, 1)) #NCHW -> NWHC
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_bn(mu, sigma, gamma, beta, binarize_input=True):
    mult = 4 if binarize_input else 2 # weights always binarized, input sometimes binarized
    #thr = ((-1*beta*torch.sqrt(sigma+1e-5) / torch.sqrt(torch.pow(gamma, 2) + 1e-4)) + mu).tolist()
    thr = ((-1*beta*torch.sqrt(sigma+1e-5) / torch.sqrt(torch.pow(gamma, 2) + 1e-5)) + mu).tolist()
    sign = [0 if elem<0 else 1 for elem in (torch.sign(gamma)).tolist()]
    return [e*mult for e in thr], sign

def convert_bn_float(mu, sigma, gamma, beta):
    return (mu*4).tolist(), (sigma*4).tolist(), gamma.tolist(), beta.tolist()

def compile_conv_block(conv_block, x, print_=True):
    conv_layer = list(conv_block.modules())[1]
    bn_layer = list(conv_block.modules())[2]

    #y = conv_block.forward(x)
    conv1_out = conv_layer(x)
    conv1_out_bn = bn_layer(conv1_out)
    y = conv1_out_bn

    x_inf = convert_conv_act(x, binarize=conv_layer.binarize_input)
    w_inf = convert_conv_weight(conv_layer.weight)
    mu, sigma, gamma, beta = bn_layer.running_mean.detach().clone(), bn_layer.running_var.detach().clone(), bn_layer.weight.detach().clone(), bn_layer.bias.detach().clone()
    bn_th_inf, bn_sign_inf = convert_bn(mu, sigma, gamma, beta, conv_layer.binarize_input)
    #bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias, conv_layer.binarize_input)
    bn_sign_inf_pack = pack(bn_sign_inf)
    y_inf = convert_conv_act(y)

    x_inf_pack = pack(x_inf)
    w_inf_pack = pack(w_inf)
    y_inf_pack = pack(y_inf)

    if print_:
        print("x", x_inf_pack, x_inf)
        print("w", w_inf_pack, w_inf)
        print("thr", bn_th_inf)
        print("sign", bn_sign_inf_pack, bn_sign_inf)
        print("y", y_inf_pack, y_inf)

    return x_inf, w_inf, bn_th_inf, bn_sign_inf, bn_sign_inf_pack, y_inf

def compile_fc_block(fc_block, x, print_=True, binarize_output=True):
    fc_layer = list(fc_block.modules())[1]
    bn_layer = list(fc_block.modules())[2]

    y = fc_block.forward(x)

    x_inf = convert_fc_act(x)
    w_inf = convert_fc_weight(fc_layer.weight)
    mu, sigma, gamma, beta = convert_bn_float(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
    bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
    bn_sign_inf_pack = pack(bn_sign_inf)
    y_inf = convert_fc_act(y, binarize=binarize_output)

    x_inf_pack = pack(x_inf)
    w_inf_pack = pack(w_inf)
    y_inf_pack = pack(y_inf)

    if print_:
        print("x", x_inf_pack, x_inf)
        print("w", w_inf_pack, w_inf)
        print("mu", mu)
        print("sigma", sigma)
        print("gamma", gamma)
        print("beta", beta)
        print("thr", bn_th_inf)
        print("sign", bn_sign_inf_pack, bn_sign_inf)
        print("y", y_inf_pack, y_inf)

    return mu, sigma, gamma, beta, bn_th_inf, bn_sign_inf, bn_sign_inf_pack, y_inf, y_inf_pack, x_inf, x_inf_pack, w_inf, w_inf_pack