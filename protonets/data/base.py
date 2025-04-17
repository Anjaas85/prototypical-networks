import torch

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]




class BalancedEpisodicSampler(EpisodicBatchSampler):
    def __init__(self, class_paths, n_way, n_episodes):  # MODIFIED INIT
        self.class_paths = class_paths
        super().__init__(len(class_paths), n_way, n_episodes)

    def __iter__(self):
        # Prevent division by zero
        class_counts = [max(len(os.listdir(d)), 1) for d in self.class_paths]
        weights = 1./torch.tensor(class_counts, dtype=torch.float32)
        
        for _ in range(self.n_episodes):
            yield torch.multinomial(weights, self.n_way, replacement=False)