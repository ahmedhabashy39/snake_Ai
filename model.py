#libraries
import torch                        
import torch.nn as nn    #neural networks 
import torch.optim as optim # optimization ex: adam
import torch.nn.functional as F # neural network evalutaoin functions as (relu, sigmoid, etc..)
import os

class Linear_QNet(nn.Module): # class linear_QNet with nn.Module as its superclass
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() #call the same intialization as the super classs
        self.linear1 = nn.Linear(input_size, hidden_size) #first layer connected with hidden layer (input,output)
        self.linear2 = nn.Linear(hidden_size, output_size)# hidden layer connected with output layer


    
    def forward(self, x): # X tensor
        x = F.relu(self.linear1(x)) #use forward relu function to rectify linear 1 
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'): #save function to save the training data 
        model_folder_path = './model'   # model_folder_path = the same path as the model file (save data in the folder as model file)
        if not os.path.exists(model_folder_path): #if the file dosn't exist in this folder
            os.makedirs(model_folder_path)# create a new path in which new_path = model_folder_path

        file_name = os.path.join(model_folder_path, file_name) #joins the file name with created path 
        torch.save(self.state_dict(), file_name) #save torch files in the obtained path


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr           #learning rate
        self.gamma = gamma      #discount rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #optimization function adam 
        self.criterion = nn.MSELoss() # define loss function as mean square error

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float) # turn the current_state list into a tensor(matrix) with float datatypes
        next_state = torch.tensor(next_state, dtype=torch.float)#turn the next_state into a tensor
        action = torch.tensor(action, dtype=torch.long) # action list to tensor
        reward = torch.tensor(reward, dtype=torch.float)# reward list to tensor
        if len(state.shape) == 1:   #if the tensor is 1 dimension only unsqueeze it to 2 dimensions
            # (1, x)
            # then unsqueeze the tensors to be two dimensional insted of 1
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # a tuple with only 1 value intialized
        # 1: predicted Q values with current state
        pred = self.model(state)  #will get a a three values predictoin for the 3 output [5.0,2,3]

        target = pred.clone() #clone the prediction
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
