
import numpy as np
import utility

class HMM:
    """
       Order 1 Hidden Markov Model
       Attributes
       ----------
       A : numpy.ndarray
           State transition probability matrix
       B: numpy.ndarray
           Output emission probability matrix with shape(N, number of output types)
       pi: numpy.ndarray
           Initial state probablity vector
       """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi


    """
        generate matrix from map
        Attributes
        ----------
        T : the length of the output
        Returns
        ----------
        hiddenStates,observationStates
        """

    def generateData(self, T):
        # 根据分布列表，返回可能返回的Index
        def _getFromProbs(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        hiddenStates = np.zeros(T, dtype=int)
        observationsStates = np.zeros(T, dtype=int)
        hiddenStates[0] = _getFromProbs(self.pi)  # 产生第一个hiddenStates
        observationsStates[0] = _getFromProbs(self.B[hiddenStates[0]])  # 产生第一个observationStates
        for t in range(1, T):
            hiddenStates[t] = _getFromProbs(self.A[hiddenStates[t - 1]])
            observationsStates[t] = _getFromProbs((self.B[hiddenStates[t]]))

        return hiddenStates, observationsStates


    def _forward(self, obseq):
        T = len(obseq)
        N = len(self.pi)

        alpha = np.zeros((T, N), dtype=float)
        alpha[0, :] = self.pi * self.B[:, obseq[0]]
        for t in range(1, T):
            for n in range(0, N):
                alpha[t, n] = np.dot(alpha[t - 1, :], self.A[:, n])  * self.B[n, obseq[t]]

        return alpha


    def viterbi(self, obseq):
        T = len(obseq)
        N = len(self.pi)
        pre_path = np.zeros((T, N), dtype=int)
        dp_matrix = np.zeros((T, N), dtype=float)
        dp_matrix[0, :] = self.pi * self.B[:, obseq[0]]

        for t in range(1, T):
            for n in range(N):
                # print(dp_matrix[t - 1, :])
                # print(self.A[:, n])
                # print(self.B[n, obseq[t]])
                probs = dp_matrix[t - 1, :] * self.A[:, n] * self.B[n, obseq[t]]
                pre_path[t, n] = np.argmax(probs)
                dp_matrix[t, n] = np.max(probs)

        print(dp_matrix)

        max_prob = np.max(dp_matrix[T-1, :])
        max_index = np.argmax(dp_matrix[T-1, :])

        path = [max_index]
        for t in reversed(range(1, T)):
            path.append(pre_path[t, path[-1]])

        path.reverse()

        return max_prob, path


    def _backward(self, obseq):
        T = len(obseq)
        N = len(self.pi)
        beta = np.zeros((T, N), dtype=float)
        beta[T - 1, :] = 1

        for t in reversed(range(T - 1)):
            for n in range(N):
                beta[t, n] = np.sum(self.A[n, :] * self.B[:, obseq[t + 1]] * beta[t + 1])

        return beta


    def baum_welch(self, obseq, criterion=0.01):
        T = len(obseq)
        N = len(self.pi)

        while(True):
            alpha = self._forward(obseq)

            beta = self._backward(obseq)

            xi = np.zeros((T - 1, N, N), dtype=float)
            for t in range(T - 1):
                denominator = np.sum(np.dot(alpha[t, :], self.A) * self.B[:, obseq[t+1]] * beta[t + 1, :])
                for i in range(N):
                    molecular = alpha[t, i] * self.A[i, :] * self.B[:, obseq[t+1]] * beta[t+1, :]
                    xi[t, i, :] = molecular / denominator

            gamma = np.sum(xi, axis=2)
            prod = (alpha[T - 1, :] * beta[T - 1, :])
            gamma = np.vstack((gamma, prod / np.sum(prod)))


            newpi = gamma[0, :]
            newA = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)
            newB = np.zeros(self.B.shape, dtype=float)

            for k in range(self.B.shape[1]):
                mask = obseq == k
                newB[:, k] = np.sum(gamma[mask, :], axis=0) / np.sum(gamma, axis=0)

            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                break

            self.A, self.B, self.pi = newA, newB, newpi




if __name__ == '__main__':

    # hiddenStates = ("Sunny", "Cloudy", "Rainy")
    # observationsStates = ("Dry", "Dryish", "Damp", "Soggy")
    
    # pi = {"Sunny": 0.63, "Cloudy": 0.17, "Rainy": 0.20}

    # A = {
    #     "Sunny": {"Sunny": 0.5, "Cloudy": 0.375, "Rainy": 0.125},
    #     "Cloudy": {"Sunny": 0.25, "Cloudy": 0.125, "Rainy": 0.625},
    #     "Rainy": {"Sunny": 0.25, "Cloudy": 0.375, "Rainy": 0.375}
    # }

    # B = {
    #     "Sunny":  {"Dry": 0.6,  "Dryish": 0.2,  "Damp": 0.15, "Soggy":0.05},
    #     "Cloudy": {"Dry": 0.25, "Dryish": 0.25, "Damp": 0.25, "Soggy":0.25},
    #     "Rainy":  {"Dry": 0.05, "Dryish": 0.10, "Damp": 0.35, "Soggy": 0.5}
    # }

    states = ('Rainy', 'Sunny')

    observations = ('walk', 'shop', 'clean')

    start_probability = {'Rainy': 0.6, 'Sunny': 0.4}

    transition_probability = {
        'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
    }

    emission_probability = {
        'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    }

    hStatesIndex = utility.generateStatesIndex(states)
    oStatesIndex = utility.generateStatesIndex(observations)
    A = utility.generateMatrix(transition_probability, hStatesIndex, hStatesIndex)
    B = utility.generateMatrix(emission_probability, hStatesIndex, oStatesIndex)
    pi = utility.generatePiVector(start_probability, hStatesIndex)
    h = HMM(A=A, B=B, pi=pi)

    observations = [0, 1, 2]
    max_prob, path = h.viterbi(observations)
    print(max_prob)
    print(path)


    alpha = h._forward(observations)
    print(alpha)

    # h.baum_welch(observations_data)
    # prob, path = h.viterbi(observations_data)
    # print(h.A)
    # print(h.B)
    # print(h.pi)