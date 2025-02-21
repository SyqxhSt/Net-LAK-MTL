import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
import math


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return


    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)

        phi_hat=0
        beta=0.5
        for g_i in pc_grad:
            #random.shuffle(grads)
            for g_j in grads:
                phi= torch.dot(g_i, g_j)/ (g_i.norm()*g_j.norm())
                if phi < phi_hat:
                    g_i=g_i+ g_j*(g_i.norm()*(phi_hat*math.sqrt(1-phi**2)-phi*math.sqrt(1-phi_hat**2))) / (g_i.norm()*math.sqrt(1-phi_hat**2))
                    phi_hat=(1-beta)*phi_hat+beta*phi


                    

       #梯度权重满足dirichlet分布
       # pweight =np.sum(np.random.dirichlet(np.ones(2), size=2),axis=0)  #size为np.ones(2)的个数
       # for i in range(2):
       #         pc_grad[i]*=pweight[i]
        u_t=[[],[]]
        u_t[0] = pc_grad[0]/pc_grad[0].norm()
        u_t[1] = pc_grad[1]/pc_grad[1].norm()

        D = pc_grad[0] - pc_grad[1]
        UT = u_t[0] - u_t[1]

        alpha_2T = pc_grad[0].matmul(UT) * (1 / (D.matmul(UT)))
        alpha = [1 - alpha_2T, alpha_2T]


        pc_grad_new = [[], []]
        pc_grad_new[0] = alpha[0] * pc_grad[0]
        pc_grad_new[1] = alpha[1] * pc_grad[1]


        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:  #reduction还原
            merged_grad[shared] = torch.stack([g[shared]  #torch.stack 沿一个新维度对张量进行连接
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')


        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad_new]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad =grads[idx]
                idx += 1
        return


    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()  #还原两个任务的梯度
            grads.append(self._flatten_grad(grad, shape))  #把每个任务梯度放在一个向量中
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads]) #cat对张量进行拼接
        return flatten_grad

    def _retrieve_grad(self): #还原梯度
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


