import numpy as np
import matplotlib.pyplot as plt
from HAC import HACClustering
from Kmeans import KMEANSClustering
from arff import Arff
from sklearn.preprocessing import MinMaxScaler


def debugDataTesting():
    mat = Arff("../data/cluster/abalone.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

    raw_data = mat.data
    data = raw_data

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    ### KMEANS ###
    KMEANS = KMEANSClustering(k=5, debug=True)
    KMEANS.fit(norm_data)
    KMEANS.save_clusters("debug_kmeans.txt")

    ### HAC SINGLE LINK ###
    HAC_single = HACClustering(k=5, link_type='single')
    HAC_single.fit(norm_data)
    HAC_single.save_clusters("debug_hac_single.txt")

    ### HAC COMPLETE LINK ###
    HAC_complete = HACClustering(k=5, link_type='complete')
    HAC_complete.fit(norm_data)
    HAC_complete.save_clusters("debug_hac_complete.txt")


def evaluationTesting():
    # label_count = 0 because clustering is unsupervised.
    mat = Arff("../data/cluster/seismic-bumps_train.arff", label_count=0)

    raw_data = mat.data
    data = raw_data

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    ### KMEANS ###
    KMEANS = KMEANSClustering(k=5, debug=True)
    KMEANS.fit(norm_data)
    KMEANS.save_clusters("evaluation_kmeans.txt")

    ### HAC SINGLE LINK ###
    HAC_single = HACClustering(k=5, link_type='single')
    HAC_single.fit(norm_data)
    HAC_single.save_clusters("evaluation_hac_single.txt")

    ### HAC COMPLETE LINK ###
    HAC_complete = HACClustering(k=5, link_type='complete')
    HAC_complete.fit(norm_data)
    HAC_complete.save_clusters("evaluation_hac_complete.txt")


def testAndPlotKmeans(norm_data, labels=False):
    k_values = []
    kmeans_sses = []

    k = 2
    while k < 8:
        print(f'Testing K = {k} ...')
        k_values += [k]

        ### KMEANS ###
        print('Testing Kmeans ...')
        KMEANS = KMEANSClustering(k=k, debug=False)
        KMEANS.fit(norm_data)
        kmeans_sse = np.sum(KMEANS.SSEs)
        kmeans_sses += [kmeans_sse]

        if all(val > 0 for val in KMEANS.cluster_sizes):
            k += 1


    print('Plotting K Means SSEs For different K values...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, kmeans_sses, label='SSE')
    for xy in zip(k_values, kmeans_sses):
        ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data')

    if labels:
        plt.title('Kmeans: SSE Values With Different K Values (With Labels)')
    else:
        plt.title('Kmeans: SSE Values With Different K Values (No Labels)')
    plt.xlabel('K Value')
    plt.ylabel('SSE Value')
    plt.legend()

    if labels:
        plt.savefig('kmeans-sse-with-labels.png')
    else:
        plt.savefig('kmeans-sse-no-labels.png')

    plt.show()


def testAndPlotSingleLink(norm_data, labels=False):
    k_values = []
    single_link_sses = []

    for k in range(2,8):
        print(f'Testing K = {k} ...')
        k_values += [k]

        ### HAC SINGLE LINK ###
        print('Testing HAC Single Link ...')
        HAC_single = HACClustering(k=k, link_type='single')
        HAC_single.fit(norm_data)
        hac_single_sse = np.sum(HAC_single.sses)
        single_link_sses += [hac_single_sse]

    print('Plotting Single Link SSEs For different K values...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, single_link_sses, label='SSE')
    for xy in zip(k_values, single_link_sses):
        ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data')
    if labels:
        plt.title('HAC Single Link: SSE Values With Different K Values (With Labels)')
    else:
        plt.title('HAC Single Link: SSE Values With Different K Values (No Labels)')

    plt.xlabel('K Value')
    plt.ylabel('SSE Value')
    plt.legend()
    if labels:
        plt.savefig('single-link-sse-with-labels.png')
    else:
        plt.savefig('single-link-sse-no-labels.png')

    plt.show()


def testAndPlotCompleteLink(norm_data, labels=False):
    k_values = []
    complete_link_sses = []

    for k in range(2, 8):
        print(f'Testing K = {k} ...')
        k_values += [k]

        ### HAC COMPLETE LINK ###
        print('Testing HAC Complete Link ...')
        HAC_complete = HACClustering(k=k, link_type='complete')
        HAC_complete.fit(norm_data)
        hac_complete_sse = np.sum(HAC_complete.sses)
        complete_link_sses += [hac_complete_sse]

    print('Plotting Complete Link SSEs For different K values...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k_values, complete_link_sses, label='SSE')
    for xy in zip(k_values, complete_link_sses):
        ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data')

    if labels:
        plt.title('HAC Complete Link: SSE Values With Different K Values (With Labels)')
    else:
        plt.title('HAC Complete Link: SSE Values With Different K Values (No Labels)')
    plt.xlabel('K Value')
    plt.ylabel('SSE Value')
    plt.legend()

    if labels:
        plt.savefig('complete-link-sse-with-labels.png')
    else:
        plt.savefig('complete-link-sse-no-labels.png')

    plt.show()


def irisDataTestingWithoutLabels():
    # label_count = 0 because clustering is unsupervised.
    mat = Arff("../data/cluster/iris.arff", label_count=1)

    raw_data = mat.data
    # Remove Labels from raw_data
    data = raw_data[:,:(raw_data.shape[1] - 1)]

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    testAndPlotKmeans(norm_data)
    testAndPlotSingleLink(norm_data)
    testAndPlotCompleteLink(norm_data)


def irisDataTestingWithLabels():
    # label_count = 0 because clustering is unsupervised.
    mat = Arff("../data/cluster/iris.arff", label_count=1)

    raw_data = mat.data
    # Remove Labels from raw_data
    data = raw_data

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    labels = True
    testAndPlotKmeans(norm_data, labels)
    testAndPlotSingleLink(norm_data, labels)
    testAndPlotCompleteLink(norm_data, labels)


def testAndPlotKmeansSameK(norm_data):
    iterations = []
    kmeans_sses = []

    k = 4
    i = 0
    while i < 5:
        print(f'Iteration {i} for K = {k} ...')
        iterations += [i]

        ### KMEANS ###
        print('Testing Kmeans ...')
        KMEANS = KMEANSClustering(k=k, debug=False)
        KMEANS.fit(norm_data)
        kmeans_sse = np.sum(KMEANS.SSEs)
        kmeans_sses += [kmeans_sse]

        if all(val > 0 for val in KMEANS.cluster_sizes):
            i += 1


    print('Plotting K Means SSEs For different K values...')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(iterations, kmeans_sses, label='SSE')
    for xy in zip(iterations, kmeans_sses):
        ax.annotate('(%s, %.2f)' % xy, xy=xy, textcoords='data')

    plt.title('Kmeans: SSE Values With K=4 & Different Initial Centroids (With Labels)')
    plt.xlabel('Iteration')
    plt.ylabel('SSE Value')
    plt.legend()
    plt.savefig('kmeans-sse-k-is-4-and-labels.png')
    plt.show()


def irisDataTestingKmeans5Times():
    mat = Arff("../data/cluster/iris.arff", label_count=1)

    raw_data = mat.data
    data = raw_data

    ### Normalize the data ###
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)

    testAndPlotKmeansSameK(norm_data)


if __name__ == '__main__':
    # debugDataTesting()
    # evaluationTesting()
    # irisDataTestingWithoutLabels()
    # irisDataTestingWithLabels()
    irisDataTestingKmeans5Times()
