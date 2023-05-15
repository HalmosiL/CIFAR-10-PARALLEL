import torch
import torch.nn as nn

def pgd_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad

    return perturbed_image

def PGD(inputs, labels, model, epsilon=8/255, stepSize=2/255, lossFun=nn.CrossEntropyLoss(), iterationNumber=3):
    adv_inputs = inputs.clone().detach().requires_grad_(True)

    outputs = model(adv_inputs)

    loss = lossFun(outputs, labels)
    loss.backward()

    data_grad = adv_inputs.grad.data
    model.zero_grad()

    for i in range(iterationNumber):
        adv_inputs = pgd_attack(adv_inputs, epsilon, data_grad)
        adv_inputs = adv_inputs.clone().detach().requires_grad_(True)

        outputs = model(adv_inputs)

        loss = lossFun(outputs, labels)
        loss.backward()

        data_grad = adv_inputs.grad.data
        model.zero_grad()

        eta = torch.clamp(adv_inputs - inputs, min=-epsilon, max=epsilon)
        adv_inputs = torch.clamp(inputs.clone() + eta, min=0, max=1).detach().requires_grad_(True)

    return adv_inputs