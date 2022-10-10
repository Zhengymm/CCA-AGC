import argparse
from sklearn.cluster import KMeans
from evaluation import eva
from module import *
from utils import *
from augmentation import *
from torch_cluster import knn_graph
from torch_geometric.utils import to_undirected


def train(model, adj, adj_aug, x, drop_feature_rate, label, epochs):

    for epoch in range(epochs):
        # get feature augmentation
        x_aug = drop_feature(x, drop_feature_rate)

        # learning representation
        h1, z1 = model(x, adj)
        h2, z2 = model(x_aug, adj_aug)

        l_h = contrastive_loss_batch(h1, h2)
        en_loss = 0.5 * l_h.mean()

        l_z = contrastive_loss_batch(z1, z2)
        pro_loss = 0.5 * l_z.mean()

        loss = args.rep * en_loss + args.pro * pro_loss
        print('Epoch [{:2}/{}]: loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        h, z = model(x, adj)
    # k-means with node representation
    kmeans = KMeans(n_clusters=label.max()+1, n_init=20)
    print(kmeans)
    cluster_z = kmeans.fit(z.data.cpu().numpy())
    y_z = cluster_z.labels_
    clu_z = eva(label, y_z, 'z-representation-kmeans')
    cluster_h = kmeans.fit(h.data.cpu().numpy())
    y_h = cluster_h.labels_
    clu_h = eva(label, y_h, 'h-representation-kmeans')

    torch.save(model.encoder.state_dict(), 'pretrain/{}_contra.pkl'.format(args.dataset))

    return clu_z, clu_h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--pro_hid', type=int, default=1024)

    parser.add_argument('--rm', type=int, default=85)
    parser.add_argument('--mask', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=0)

    parser.add_argument('--rep', type=float, default=1)
    parser.add_argument('--pro', type=float, default=1)

    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--project', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading PyG datasets
    dataset = get_dataset('data', args.dataset)
    print(dataset)
    data = dataset[0]    # data.x, data.edge_index, data.y
    features = data.x
    labels = data.y.numpy()
    n_cluster = labels.max() + 1

    features = features.to(device)
    edge_index = data.edge_index.to(device)
    n = data.num_nodes

    rm_edge = args.rm
    feature_drop = args.mask

    # augmentation
    if args.k == 0:
        knn_edge = data.edge_index
    else:
        savepath = 'data/load_data'
        if os.path.exists('{}/{}_{}nn.npy'.format(savepath, args.dataset, args.k)):
            knn_edge = np.load('{}/{}_{}nn.npy'.format(savepath, args.dataset, args.k), allow_pickle=True)
        else:
            knn_edge = knn_graph(data.x, args.k)
            np.save('{}/{}_{}nn'.format(savepath, args.dataset, args.k), knn_edge)

    aug_edge = sample_graph_own(args.dataset, data.x, knn_edge, rm_edge, k=args.k)
    edge_index_aug = torch.from_numpy(aug_edge.T)

    edge_index_aug = to_undirected(edge_index_aug).to(device)

    # model
    encoder = Encoder(in_channels=dataset.num_features, out_channels=args.out_dim,
                      hidden=args.hidden, activation=args.activation)

    model = Contra(encoder=encoder,
                   hidden_size=args.out_dim,
                   projection_hidden_size=args.pro_hid,
                   projection_size=args.pro_hid,
                   n_cluster=n_cluster).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    clu_z, clu_h = train(model, edge_index, edge_index_aug, features, feature_drop, labels, args.epochs)

