# -*- coding: utf-8 -*-
# @Time   : 2022/5/13
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
SSCDR
################################################
Reference:
    SeongKu Kang et al. "Semi-Supervised Learning for Cross-Domain Recommendation to Cold-Start Users" in CIKM 2019.
"""
import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class SSCDR(CrossDomainRecommender):
    r"""SSCDR conducts the embedding mapping by both supervised way and semi-supervised way.
        In this implementation, the mapped embedding is used for all the overlapped users (or items) in target domain.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SSCDR, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "SSCDR model only support user overlapped or item overlapped dataset!"
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = None
        self.dataset = dataset.source_domain_dataset.inter_feat
        # load dataset info
        self.embedding_size = config['embedding_size']
        self.lamda = config['lambda']
        self.margin = config['margin']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.mapping_layer = MLPLayers(layers=[self.embedding_size] + self.mlp_hidden_size + [self.embedding_size],
                                            activation='tanh', dropout=0, bn=False)
        if self.mode == 'overlap_users':
            self.user_interacted_items = self.build_interacted_items(dataset, mode='user')
            # This means that we want to build an interacted item for each user in our dataset (the list of items).
            # We do this by calling the function self.build_interacted_items() with two arguments: data and mode="user".
            # This function returns a list of tuples where each tuple has three values: item ID, user ID, and interaction count (0 if no interaction).
        elif self.mode == 'overlap_items':
            self.item_interacted_users = self.build_interacted_items(dataset, mode='item')
            # This function returns a list of items and users who interacted with each other, respectively.

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(self.total_num_users, self.embedding_size)
        # This is a function that creates an embedding matrix for the users in the source domain. The embedding size is defined by the user and it's usually set to be between 50 and 100.
        # The input is a list of users in the source domain. The output is an embedding matrix for each user.
        self.source_item_embedding = torch.nn.Embedding(self.total_num_items, self.embedding_size)
        # This is a function that creates an embedding matrix for the items in the source domain. The embedding size is defined by the user and it's usually set to be between 50 and 100. The input is a list of items in the source domain. The output is an embedding matrix with dimensions (total number of items, embedding size).

        self.target_user_embedding = torch.nn.Embedding(self.total_num_users, self.embedding_size)
        self.target_item_embedding = torch.nn.Embedding(self.total_num_items, self.embedding_size)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.map_loss = nn.MSELoss()
        self.rec_loss = nn.TripletMarginLoss(margin=self.margin)


        """for above code explanation is given below:
        The code starts by creating a SSCDR object.The _init_() method takes two arguments: a configuration object and a dataset. The first argument is the configuration object. This object contains information about how to initialize the SSCDR model.
        In this case, the only required parameter is the dataset name. The second argument is the dataset. This is where you will store your data.
        You can either store your data in a single file or you can split it up into multiple files. If you want to use user overlapped data, then you need to specify that as an option when initializing the SSCDR model.
        If you want to use item overlapped data, then you don't need to specify anything since that's automatically handled by the SSCDR model. After initialization is complete, the code sets up some basic properties of the SSCDR model. 
        First, it defines whether or not the SSCDR model should work with user or item overlapped datasets (mode). Next, it sets up a phase variable which will track which stage of analysis is currently active (i.e., pre-processing or prediction).
        Finally, it creates an instance of NumPy for use.The code initializes an instance of the SSCDR class. The _init_() method takes two arguments: a configuration object and a dataset.
        The first argument is an object that contains information about the model. The second argument is a data set. If the dataset has more than one item, then the SSCDR class will operate in "overlap_users" mode.
        If the dataset has more than one user, then the SSCDR class will operate in "overlap_items" mode. Otherwise, the SSCDR class will operate in "non_overlap" mode.The phase attribute stores information about when to start and stop data processing for this model. This attribute is None by default. The code first creates a new instance of the MLPLayers class.
        This class is used to create a data structure that can be used to train and predict models. The first parameter passed into the constructor of the MLPLayers class is the number of layers in the data structure.
        In this case, there are three layers: one for encoding feature values, one for mapping these values to target features, and one for predicting target values. Next, the code sets up some configuration variables.
        The first variable is self.embedding_size, which specifies how many feature values will be stored in each layer of the data structure. The second variable is self.mlp_hidden_size, which specifies how many hidden units will be used in each layer of the data structure (i.e., how many unique instances of each target feature will be included).
        Finally, self.mapping_layer defines which layer will contain the mapping function between input features and target features (in this case it's set to be equal to self.embedding_size + self.mlp_hidden_size). Now that all of these configuration variables have been set up, it's time to start training and predicting models!
        To do this, the code initializes a new instance of the MLPLayers class, specifying the size of the embedding layer (self.embedding_size), the size of the hidden layer (self.mlp_hidden_size), and an additional layer for mapping features to labels (self.mapping_layer).
        The MLPLayers class also specifies that it will require three layers: one for encoding data, one for decoding data, and one for mapping features to labels. The code first creates a list of items and users who interacted with each other.
        The code then uses the function build_interacted_items() to create the list of items that were interacted with by users, and the function build_interacted_items() to create the list of users who interacted with items.
        The code also defines two modes: user and item. In user mode, the code builds a list of users who interacted with items; in item mode, it builds a list of users who interacted with each other.
        Finally, the code calls two functions: one to return a list of items that were interacted with by users, and another to return a list of users who interacted with each other. The code will first check the mode of the code block.
        If it is set to 'overlap_users', then the function will create a list of users who interacted with each other. If the mode of the code block is set to 'overlap_items', then the function will create a list of items that were interacted with by users.
        The code defines three layers: a source layer, an item layer, and a target layer. The source layer contains the number of users and items in the data set. The item layer contains the embedding size for each user and item. The target layer contains the number of users and items after applying a loss function to reduce bias in the data set.
        The code first defines two embeddings: a source embedding and a target embedding. Then it creates an Embedding object for each of these layers using the nn module from Torch7. Finally, it sets up a with statement that uses no_grad() to create an analyzer that reduces bias in the data set by adjusting its weights according to a loss function.
        The code defines three layers - source, item, and target. The source layer will contain the number of users and items, while the item layer will contain the number of users and items per layer.
        The target layer will contain the total number of users and items. The next line creates an embedding for each layer. The source embedding will use the number of users and items in the source layer, while the item embedding will use the number of users and items per layer.
        Finally, a no_grad() call is made to prevent Gradient Descent from being applied to these layers."""

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def build_interacted_items(self, dataset, mode='user'):
        dataset = dataset.source_domain_dataset
        if mode == 'user':
            interacted_items = [[] for _ in range(self.total_num_users)]
            for uid, iid in zip(dataset.inter_feat[dataset.uid_field].numpy(),
                                dataset.inter_feat[dataset.iid_field].numpy()):
                interacted_items[uid].append(iid)
            return interacted_items
        else:
            interacted_users = [[] for _ in range(self.total_num_items)]
            for iid, uid in zip(dataset.inter_feat[dataset.iid_field].numpy(),
                                dataset.inter_feat[dataset.uid_field].numpy()):
                interacted_users[iid].append(uid)
            return interacted_users
            # if mode == 'user': then it will create an empty list called interacted_items and add each item from dataset into this list one at a time until there are no more items left in dataset.
            # The code is used to build a list of items that have been interacted with by the user.
            # The code starts by creating a dataset object.
            # The dataset is the source domain of the data that will be used in this code.
            # Next, it creates an interacted_users list which contains all of the users who have been interacted with.
            # Then, for each user in the dataset, it appends them to the list of users that have been interacted with.
            # Finally, it returns all of these lists as one big list called "interacted_items".
            # The code is used to build a list of items that have been interacted with by the user.
            # This function returns a list of tuples where each tuple has three values: item ID, user ID, and interaction count (0 if no interaction).

    def sample(self, ids, mode='user'):
        ids = ids.cpu().numpy()
        interacted = np.zeros_like(ids)
        non_interacted = np.zeros_like(ids)
        if mode =='user':
            all_candidates = list(range(self.overlapped_num_items)) + \
                             list(range(self.target_num_items, self.total_num_items))
            for index, id in enumerate(ids):
                interacted_items = self.user_interacted_items[id]
                if len(interacted_items) == 0:
                    interacted_items.append(0)
                non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                while non_interacted_id in interacted_items:
                    non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                interacted[index] = np.random.choice(interacted_items, size=1)[0]
                non_interacted[index] = non_interacted_id
        else:
            all_candidates = list(range(self.overlapped_num_users)) + \
                             list(range(self.target_num_users, self.total_num_users))
            for index, id in enumerate(ids):
                interacted_users = self.item_interacted_users[id]
                if len(interacted_users) == 0:
                    interacted_users.append(0)
                non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                while non_interacted_id in interacted_users:
                    non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                interacted[index] = np.random.choice(interacted_users, size=1)[0]
                non_interacted[index] = non_interacted_id
        return torch.from_numpy(interacted).to(self.device), torch.from_numpy(non_interacted).to(self.device)
        # This is a function that samples users and items from the source domain. The input is a list of user IDs or item IDs in the source domain. The output is two lists: one with interacted users/items and another with non-interacted users/items.

    @staticmethod
    def embedding_normalize(embeddings):
        emb_length = torch.sum(embeddings**2, dim=1, keepdim=True)
        ones = torch.ones_like(emb_length)
        norm = torch.where(emb_length > 1, emb_length, ones)
        return embeddings / norm
        # This is a function that normalizes the embedding matrix. The input is an embedding matrix, and the output is a normalized embedding matrix.
        # The function first calculates the length of each row in the embedding matrix. Then, if any value is greater than 1, it replaces that value with 1. Finally, it divides each element in the embedding matrix by its corresponding length to normalize all elements to be between 0 and 1.

    @staticmethod
    def embedding_distance(emb1, emb2):
        return torch.sum((emb1-emb2)**2, dim=1)
        # This is a function that calculates the distance between two embeddings. 
        # The input is two embedding matrices and the output is a list of distances between each pair of users in the source domain and target domain.

    def set_phase(self, phase):
        self.phase = phase

    def calculate_source_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_pos_item = interaction[self.SOURCE_ITEM_ID]
        source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

        source_user_e = self.source_user_embedding(source_user)
        source_pos_item_e = self.source_item_embedding(source_pos_item)
        source_neg_item_e = self.source_item_embedding(source_neg_item)

        loss_t = self.rec_loss(self.embedding_normalize(source_user_e),
                               self.embedding_normalize(source_pos_item_e),
                               self.embedding_normalize(source_neg_item_e))
        return loss_t
        # This is a function that calculates the loss for the source domain. The input is a list of interactions between users and items in the source domain. 
        # The output is the loss value.

    def calculate_target_loss(self, interaction):
        target_user = interaction[self.TARGET_USER_ID]
        target_pos_item = interaction[self.TARGET_ITEM_ID]
        target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

        target_user_e = self.target_user_embedding(target_user)
        target_pos_item_e = self.target_item_embedding(target_pos_item)
        target_neg_item_e = self.target_item_embedding(target_neg_item)

        loss_t = self.rec_loss(self.embedding_normalize(target_user_e),
                               self.embedding_normalize(target_pos_item_e),
                               self.embedding_normalize(target_neg_item_e))
        return loss_t
        # This function calculates the loss of the target domain. It takes in a list of interactions and returns a loss value. 
        # The interaction is a list of tuples where each tuple has three values: user ID, item ID, and interaction count (0 if no interaction).
        # The function first defines the target user, positive item and negative item. 
        # Then, it calculates the embedding of each user and item using self.target_user_embedding() and self.target_item_embedding(). Finally, it returns a loss value by calling rec_loss().

    def calculate_map_loss(self, interaction):
        idx = interaction[self.OVERLAP_ID].squeeze(1)
        if self.mode == 'overlap_users':
            source_user_e = self.source_user_embedding(idx)
            target_user_e = self.target_user_embedding(idx)
            map_e = self.mapping_layer(source_user_e)
            loss_s = self.map_loss(map_e, target_user_e)
            source_pos_item, source_neg_item = self.sample(idx, mode='user')

            map_pos_item_e = self.mapping_layer(self.source_item_embedding(source_pos_item))
            map_neg_item_e = self.mapping_layer(self.source_item_embedding(source_neg_item))
            loss_u = self.rec_loss(self.embedding_normalize(target_user_e),
                                   self.embedding_normalize(map_pos_item_e),
                                   self.embedding_normalize(map_neg_item_e))
        else:
            source_item_e = self.source_item_embedding(idx)
            target_item_e = self.target_item_embedding(idx)
            map_e = self.mapping_layer(source_item_e)
            loss_s = self.map_loss(map_e, target_item_e)
            source_pos_user, source_neg_user = self.sample(idx, mode='item')

            map_pos_user_e = self.mapping_layer(self.source_user_embedding(source_pos_user))
            map_neg_user_e = self.mapping_layer(self.source_user_embedding(source_neg_user))
            loss_u = self.rec_loss(self.embedding_normalize(target_item_e),
                                   self.embedding_normalize(map_pos_user_e),
                                   self.embedding_normalize(map_neg_user_e))
        return loss_s + self.lamda * loss_u
        # This is a function that calculates the loss of the model. The loss is calculated by two parts: supervised learning and semi-supervised learning. 
        # The supervised learning part is to calculate the difference between the embedding of users or items in source domain and target domain, while semi-supervised learning part is to calculate the difference between user-item pairs in source domain and target domain.
        # The code calculates the loss for an interaction between two users.
        # The first line of code calculates the index of the user in question, which is then used to calculate a source and target embedding layer from that user.
        # The next line calculates the map loss for this interaction by multiplying together both layers.
        # The code calculates the map loss of an interaction between two users.
        # The first line of code squeezes the index of the OVERLAP_ID variable to only have one element, idx.
        # The next line checks if self.mode == 'overlap_users'.
        # If so, it will calculate the source user embedding and target user embedding for this interaction before calculating the map loss.

    def calculate_loss(self, interaction):
        if self.phase == 'SOURCE':
            return self.calculate_source_loss(interaction)
        elif self.phase == 'OVERLAP':
            return self.calculate_map_loss(interaction)
        else:
            return self.calculate_target_loss(interaction)
            # This is a function that calculates the loss of the model. The input is an interaction between users and items, and the output is a scalar value representing the loss.
            # If self.phase == 'SOURCE': return self.calculate_source_loss(interaction). This means that if we're in the source domain, then calculate the loss of this model using function calculate_source_loss().
            # elif self.phase == 'OVERLAP': return self.calculate_map_loss(interaction). This means that if we're in overlap phase, then calculate the loss of this model using function calculate_map_loss().
            # else: return self.calculate_target__oss(interaction). This means that if we're not in either source or overlap phase, then calculate the loss of this model using function calulate target__oss().

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            user_e = self.embedding_normalize(self.source_user_embedding(user))
            item_e = self.embedding_normalize(self.source_item_embedding(item))
            score = -self.embedding_distance(user_e, item_e)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self.embedding_normalize(self.target_user_embedding(user))
            item_e = self.embedding_normalize(self.target_item_embedding(item))
            score = -self.embedding_distance(user_e, item_e)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.embedding_size, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping_layer(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                item_e = self.target_item_embedding(item)
            else:
                user_e = self.target_user_embedding(user)
                repeat_item = item.repeat(self.embedding_size, 1).transpose(0, 1)
                item_e = torch.where(repeat_item < self.overlapped_num_items, self.mapping_layer(self.source_item_embedding(item)),
                                     self.target_item_embedding(item))
            user_e = self.embedding_normalize(user_e)
            item_e = self.embedding_normalize(item_e)
            score = -self.embedding_distance(user_e, item_e)
        return score
        # This is a function that predicts the score of each user-item pair. The input is a list of user-item pairs, and the output is a list of scores.
        """First, the function checks whether the phase is source or target. If so, it will use the embedding of users and items in that domain to calculate their score.
        Otherwise, if there are overlapped users or items between two domains (i.e., mode = 'overlap_users' or 'overlap_items'), then we need to map the embeddings from source domain to target domain by using a mapping layer (mapping_layer).
        The mapped embedding is used for all the overlapped users (or items) in target domain. Finally, we normalize user and item embeddings before calculating their scores with dot product distance method (-self.embedding_distance(user_e, item_e))."""

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.embedding_normalize(self.source_user_embedding(user))
            overlap_item_e = self.embedding_normalize(self.source_item_embedding.weight[:self.overlapped_num_items])
            source_item_e = self.embedding_normalize(self.source_item_embedding.weight[self.target_num_items:])
            all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.embedding_normalize(self.target_user_embedding(user))
            all_item_e = self.embedding_normalize(self.target_item_embedding.weight[:self.target_num_items])
        else:
            user = interaction[self.TARGET_USER_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.embedding_size, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping_layer(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            else:
                user_e = self.target_user_embedding(user)
                overlap_item_e = self.mapping_layer(self.source_item_embedding.weight[:self.overlapped_num_items])
                target_item_e = self.target_item_embedding.weight[self.overlapped_num_items:self.target_num_items]
                all_item_e = torch.cat([overlap_item_e, target_item_e], dim=0)
            user_e = self.embedding_normalize(user_e)
            all_item_e = self.embedding_normalize(all_item_e)

        num_batch_user, emb_dim = user_e.size()
        num_all_item, _ = all_item_e.size()
        dist = -2 * torch.matmul(user_e, all_item_e.permute(1, 0))
        dist += torch.sum(user_e ** 2, -1).view(num_batch_user, 1)
        dist += torch.sum(all_item_e ** 2, -1).view(1, num_all_item)
        return -dist.view(-1)
        # This is a function that predicts the interaction between users and items. It takes in an interaction (a tuple of user ID and item ID) as input, and returns the predicted score for this interaction. 
        # The higher the score, the more likely it is that this user will interact with this item. 
