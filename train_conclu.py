import argparse
from sklearn.cluster import KMeans
from evaluation import eva
from module import *
from utils import get_dataset
from augmentation import *
from torch_cluster import knn_graph
from torch_geometric.utils import to_undirected


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train(model, adj, adj_aug, x, drop_feature_rate, label, epochs):
    clu = []

    for epoch in range(epochs):
        model.train()

        x_aug = drop_feature(x, drop_feature_rate)

        h1, z1 = model(x, adj)
        h2, z2 = model(x_aug, adj_aug)
        q1, q2 = model.kl_cluster(h1, h2)

        q1_pred = q1.detach().cpu().numpy().argmax(1)
        clu_q1 = eva(label, q1_pred, 'Q1_self_cluster', True)

        if epoch % args.update_p == 0:
            p1 = target_distribution(q1.data)
            p_pred = p1.detach().cpu().numpy().argmax(1)
            eva(label, p_pred, 'P_self_cluster', True)

        kl1 = F.kl_div(q1.log(), p1, reduction='batchmean')
        kl2 = F.kl_div(q2.log(), p1, reduction='batchmean')
        con = F.kl_div(q2.log(), q1, reduction='batchmean')
        clu_loss = kl1 + kl2 + con

        l_h = contrastive_loss_batch(h1, h2)
        en_loss = 0.5 * l_h.mean()

        l_z = contrastive_loss_batch(z1, z2)
        pro_loss = 0.5 * l_z.mean()

        loss = args.rep * en_loss + args.pro * pro_loss + args.clu * clu_loss

        clu.append(clu_q1)
        print('Epoch [{:2}/{}]: loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return clu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--out_dim', type=int, default=512)
    parser.add_argument('--pro_hid', type=int, default=1024)

    parser.add_argument('--rm', type=int, default=65)
    parser.add_argument('--mask', type=float, default=0.4)
    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--rep', type=float, default=1)
    parser.add_argument('--clu', type=float, default=1)
    parser.add_argument('--pro', type=float, default=1)
    parser.add_argument('--update_p', type=int, default=1)

    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0)

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
    edges = np.array(data.edge_index)
    n = data.num_nodes

    rm_edge = args.rm
    feature_drop = args.mask

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
                      hidden=args.hidden, activation=args.activation).to(device)

    model = Contra(encoder=encoder,
                   hidden_size=args.out_dim,
                   projection_hidden_size=args.pro_hid,
                   projection_size=args.pro_hid,
                   n_cluster=n_cluster,
                   n_node=n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # load pre-train for clustering initialization
    save_model = torch.load('pretrain/{}_contra.pkl'.format(args.dataset), map_location='cpu')

    model.encoder.load_state_dict(save_model)
    with torch.no_grad():
        h_o, z_o = model(features, edge_index)
    kmeans = KMeans(n_clusters=n_cluster, n_init=20)
    clu_pre = kmeans.fit_predict(h_o.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(labels, clu_pre, 'Initialization')

    clu_acc = train(model, edge_index, edge_index_aug, features, feature_drop, labels, args.epochs)

    clu_q_max = np.max(np.array(clu_acc), 0)
    clu_q_final = clu_acc[-1]


