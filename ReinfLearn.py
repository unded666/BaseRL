import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt


class ERLearn:
    def __init__(self, baseline = 60):
        self.__states = None
        self.__actions = None
        self.__Func = None
        self.__Q = None
        self.__alpha = 0.1
        self.__gamma = 0.1
        self.__epsilon = 0.05
        self.__history = None
        self.__max_steps = 200
        self.__Fun_reset = None
        self.__history = pd.DataFrame()
        self.__t_states = None
        self.__metrics = {}
        self.__fetch_state = None
        self.__reset_value = 100
        self.__constrained_action_space = False
        self.__get_val_act = None
        self.__baseline_init = baseline

    @property
    def get_val_act(self):
        return self.__get_val_act

    @get_val_act.setter
    def get_val_act(self, Fn):
        self.__get_val_act = Fn

    @property
    def constrained_action_space(self):
        return self.__constrained_action_space

    @constrained_action_space.setter
    def constrained_action_space(self, constraint):
        self.__constrained_action_space = constraint

    @property
    def reset_value(self):
        return self.__reset_value

    @reset_value.setter
    def reset_value(self, R):
        self.__reset_value = R

    @property
    def fetch_state(self):
        return self.__fetch_state

    @fetch_state.setter
    def fetch_state(self, s):
        self.__fetch_state = s

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, m):
        self.__metrics = m

    @property
    def max_steps(self):
        return self.__max_steps

    @max_steps.setter
    def max_steps(self, M):
        self.__max_steps = M

    @property
    def history(self):
        return self.__history

    @history.setter
    def history(self, h):
        self.__history = h.copy()  # any history being imported should be a dataframe

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, E):
        self.__epsilon = E

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, a):
        self.__alpha = a

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, g):
        self.__gamma = g

    def assign_reset(self, func):
        self.__Fun_reset = func

    def reset_func(self, *args):
        self.__Fun_reset(*args)

    def reset_history(self):
        """
        resets both the history and the measurement metrics to be empty
        """
        self.__history = pd.DataFrame()
        self.__metrics = {}

    def set_states(self, S):
        # Removes duplicates from any assigned states

        self.__states = [list(set(s)) for s in S]
        # convert states to strings for column references
        self.__states = [[str(substring) for substring in string_i] for string_i in S]


    def expose_states(self):

        return self.__states

    def set_actions(self, A):

        # set actions to strings for safe table creation

        self.__actions = [str(a) for a in A]

    def expose_actions(self):

        return self.__actions

    def compile_Qspace(self):

        if self.__states and self.__actions:

            Qcols = list(itertools.product(*self.__states))
            vals = np.random.rand(len(self.__actions), len(Qcols))*self.__baseline_init
            Q = pd.DataFrame(vals, columns=Qcols, index=self.__actions)

            # re-initialise __Q in case of errors in columns
            self.__Q = pd.DataFrame()

            self.__Q = Q.copy()
        else:
            if not self.__actions:
                print("Action space needs to be defined")
            if not self.__states:
                print("State space needs to be defined")

    def expose_Qtable(self):

        R = None
        if self.__Q is not None:
            R = self.__Q

        return R

    def reset_table(self):
        self.compile_Qspace()
        self.set_terminal_states(self.__t_states)

    def set_terminal_states(self, terminal_states):
        """
        sets all of the terminal states to be zero return values, as in those
        states the value being returned will be the reward value instead

        I need to do away with this logic, as it's not common enough for a generic
        RL solver. A boolean 'Done' response from the step method is more generalisable
        """

        if self.__Q is not None:
            for t_state in terminal_states:
                self.__Q[t_state] = 0
            self.__t_states = terminal_states
        else:
            print("Q-table still needs to be initialised")

    def assign_func(self, given):
        """
        Assigns a function to the RL object, such that when given an action and a state produces a return
        and a new state.
        """

        self.__Func = given

    ##########################################
    ### ab_choice tie-in methods #############
    ##########################################

    def update_optimiser(self, S0, A0, R, done, S1=None):

        Q0 = self.__Q.loc[[A0], S0].values
        if not done:
            A1 = self.get_optimal_parameters (S1)
            Q1 = self.__Q.loc[[A1], S1].values
        else:
            Q1 = 0
            if R == 0:
                bunny = 5
        Qnew = Q0 + self.__alpha * (R + self.__gamma * Q1 - Q0)
        self.__Q.loc[[A0], S0] = Qnew


    def get_optimal_parameters(self, state):

        best_action = self.__actions[np.argmax(self.__Q.loc[:, [state]].values)]

        return best_action

    ##########################################

    def run_func(self, S, *args):

        return self.__Func(S, *args)

    def __QA(self, S, a):

        return self.__Q.loc[a, S]

    def __max_QA(self, S, A):

        QA = [(self.__Q.loc[a, [S]][0], a) for a in A]
        Q, A = zip(*QA)
        return max(Q), A[np.argmax(Q)]

    def update_Q(self, s0, s1, a0, a1, R):
        """
        updates Q (S0, A0) according to the update rule for either Q-learning or SARSA learning.
        Q / SARSA is determined by the parameter passed into a1, with a1 = policy(S0, a0) being
        SARSA and a1 = argmax(S0, a0) being Q-learning

        Ignore the above, it is a lie. Some refactoring will be required to
        implement SARSA, this implements Q-Learning
        """

        Q0 = self.__Q.loc[[a0], [s0]].values
        # if statement negates the need for setting terminal states to zero.
        # if not R == 0:
        if R == 0:
            Q1 = self.__Q.loc[[a1], [s1]].values
        else:
            Q1 = 0
        Qnew = Q0 + self.__alpha * (R + self.__gamma * Q1 - Q0)
        self.__Q.loc[[a0], [s0]] = Qnew

    def __get_valid_actions(self, A):

        val_acts = self.__get_val_act(A)

        return val_acts

    def policy_choice(self, s):

        ex = 0
        R = np.random.random()
        if not self.__constrained_action_space:
            A = self.__actions.copy()
        else:
            A = self.__get_valid_actions(self.__actions.copy())
        Q, a = self.__max_QA(s, A)
        if R <= self.__epsilon:  # exploration
            if (
                len(A) > 1
            ):  # do not remove choice if the choice is the only viable choice in town
                A.remove(a)
                ex = 1
            a = np.random.choice(A)
            if type(a) == np.str_: a = str(a) #numpy.random.choice converts string types for no good reason, so fix it
        else:  # greedy, keep a
            pass

        return a, ex

    def do_stuff(self, s):
        """
        Performs one iteration of the Q-learning algorithm, utilising the
        member functions in terms of the Q-learning abstractions. Requires
        all of the relevant compilations to be done first (maybe should have an
        audit function? could be very slow, better to audit only in the loop)
        """
        # Choose a0 from A according to policy under s0
        a0, ex = self.policy_choice(s)
        # take action a0, observe R and s1
        # R, s1, done = self.run_func (a0, s, ex)
        R, s1, done = self.run_func(a0, s)
        # find a1
        a1 = self.__actions[np.argmax(self.__Q.loc[:, [s1]].values)]
        # update Q(s0,a0) according to update rule which will determine a1
        self.update_Q(s, s1, a0, a1, R)

        return a0, s1, R, done

    def run_episode(self, verbose):

        """
        runs one full episode from initialisation to end of episode, logging results
        for future inspection
        """

        hist_cols = ["timed_out", "steps_taken", "Reward"]

        # initialise state and environment for new episode
        self.reset_func(self.reset_value)
        s, _ = self.fetch_state()

        if verbose:
            print(f"starting point :{s}")
        i = 0  # index for counter
        done = False
        while not done:
            i += 1
            a, s, R, done = self.do_stuff(s)
            # log results
            # Fix me!! R should be able to give a negative (or zero) reward without timing out.
            # Disambiguate reward and timing!
            if i < self.__max_steps:
                timed_out = False
            else:
                timed_out = True
            hist_i = (timed_out, i, R)
            if (
                i > self.__max_steps or not R == 0
            ):  # could be a problem if the genuine reward is zero, might need to explore a done boolean parameter
                done = True
        # self.__history = self.__history.append (dict(zip(hist_cols, hist_i)), ignore_index = True)
        cur_hist = pd.DataFrame()
        cur_hist = cur_hist.append(dict(zip(hist_cols, hist_i)), ignore_index=True)
        parms = {
            "alpha": self.__alpha,
            "gamma": self.__gamma,
            "epsilon": self.__epsilon,
        }
        for P in parms:
            cur_hist[P] = parms[P]
        self.__history = self.__history.append(cur_hist, ignore_index=True)

        return i, R

    def __get_individual_runs(self):

        H = self.__history.copy()
        remove_me = {"steps_taken", "timed_out", "Reward"}
        C = [c for c in H.columns if not c in remove_me]
        H["combo"] = list(zip(*[H[c] for c in C]))
        categs = H["combo"].unique()
        Hi = {}
        for cat in categs:
            Hi[cat] = H[H["combo"] == cat]
            Hi[cat] = Hi[cat].drop(["combo"], axis=1)

        return Hi

    def __pc_success_rate(self, Hi, Convtime, bound):
        """
        returns the post-convergence success rate of the experiment,
        being the percentage of the attempts that performed within the
        given boundary.
        """
        vals = Hi["Reward"].values[Convtime::]
        M = np.mean(vals >= bound)
        M = M * 100

        return M

    def __convergence_time(self, Hi, bound, window=5, leeway=0.5):

        ubound = bound * (1 - leeway)
        V = Hi["Reward"].rolling(window).mean().values

        T = next((i for i, v in enumerate(V) if v >= ubound))

        return T

    def __stable_performance(self, Hi, Convtime):

        vals = Hi["Reward"].values[Convtime::]
        M = np.mean(vals)
        M = M

        return M

    def display_metrics(self, bound):
        def expand_name(given):

            metrics = ["a", "g", "e"]
            O = ""
            for i, G in enumerate(given):
                O = O + f"{metrics[i]} = {G}" + ", "

            O = O[:-2]
            return O

        self.eval_metrics(bound)
        titles = ["convergence time", "success rate pc", "stable performance"]
        for i, title in enumerate(titles):
            plt.subplot(1, 3, i + 1)
            plt.title(title)
            y = list(self.__metrics.loc[:, title].values)
            xtck = list(self.__metrics.index)
            xtcks = [expand_name(xt) for xt in xtck]
            # xtcks = ['{} = {}'.format(metrics[i], xt) for i, xt in enumerate(xtck)]
            x = range(len(y))
            plt.bar(
                x,
                y,
                color=[
                    "blue",
                    "orange",
                    "green",
                    "red",
                    "black",
                ],
            )
            plt.xticks(x, xtcks, rotation=30, fontsize=4)

        return None

    def eval_metrics(self, opt_bound):

        H = self.__get_individual_runs()
        met_df = pd.DataFrame(
            columns=["convergence time", "success rate pc", "stable performace"],
            index=list(H.keys()),
        )
        for Hi in H:
            # determine success benchmark? (just use opt_bound for now)
            bench = opt_bound

            # time to convergence
            T = self.__convergence_time(H[Hi], bench)
            met_df.loc[Hi, "convergence time"] = T
            # % success after convergence
            met_df.loc[Hi, "success rate pc"] = self.__pc_success_rate(H[Hi], T, bench)
            # stable permance
            met_df.loc[Hi, "stable performance"] = self.__stable_performance(H[Hi], T)

        self.__metrics = met_df

    def display_history(self, benchmark=4, **kwargs):

        H = self.__history.copy()
        remove_me = {"steps_taken", "timed_out", "Reward"}
        C = [c for c in H.columns if c not in remove_me]
        H["combo"] = list(zip(*[H[c] for c in C]))
        categs = H["combo"].unique()
        Leg = []
        max_x = 0
        for cat in categs:
            Hi = H[H["combo"] == cat]
            lg = "a = {}, g = {}, e = {}".format(
                Hi["alpha"].values[0], Hi["gamma"].values[0], Hi["epsilon"].values[0]
            )
            Leg.append(lg)
            max_x = np.max([max_x, Hi.shape[0]])
            x = range(Hi.shape[0])
            plt.plot(x, Hi["Reward"])
            del Hi

        for kw in kwargs:
            x = range(max_x)
            y = np.ones(max_x) * kwargs[kw]
            plt.plot(x, y)
            Leg.append(kw)
        # best = np.ones(len(x)) * benchmark
        plt.grid()
        plt.legend(Leg)
        plt.title(f"Learning over {len(categs)} parametric combinations")
        plt.xlabel("learning attempt")
        plt.ylabel("steps to completion")
        plt.show()

        # Old version here. New version below allows for multiple plot on same
        # axis, one plot for each unique combination of parameters
        """
        L = self.__history.shape[0]
        x = range (L)
        best = np.ones(L)*benchmark
        plt.plot (x, self.__history['steps_taken'], x, best)
        plt.grid()
        plt.title ('Learning over {} steps\nalpha = {}\ngammma = {}\nepsilon = {}'.format(L, self.__alpha, self.__gamma, self.__epsilon))
        plt.xlabel ('learning attempt')
        plt.ylabel ('steps to completion')
        plt.legend (['RL Performance','Optimality bound'])
        """

        return None

    def run_simulation(self, n, verbose=True):

        # initialise history logs
        hist_cols = ["timed_out", "steps_taken", "Reward"]
        # self.__history = pd.DataFrame (columns = hist_cols)
        i = 0
        for _ in range(n):
            i += 1
            N, Ri = self.run_episode(verbose)
            if verbose:
                print(f"episode {i} finished in {N} steps")

        if verbose:
            print(f"finished running {n} episodes")


if __name__ == "__main__":
    # debugging code here
    print("I am debugging code! hear me debug!")
    from Environments import THunt
    RL = ERLearn()
    TH = THunt()
    A = [0, 1, 2, 3]
    SSpace = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    RL.set_actions(A)
    RL.set_states(SSpace)
    THunt.treasure = (3, 3)
    RL.assign_func(TH.take_step)
    RL.assign_reset(TH.reset_prize)
    RL.fetch_state = TH.get_current_state
    RL.compile_Qspace()
    RL.set_terminal_states([TH.treasure])
    RL.alpha = 0.1
    RL.gamma = 0.1
    RL.epsilon = 0.1
    # hard-coding debugging

    #RL.reset_func(RL.reset_value)
    #s, _ = RL.fetch_state()
    #a0, ex = RL.policy_choice(s)
    # take action a0, observe R and s1
    # R, s1, done = self.run_func(a0, s, ex)
    RL.run_simulation(100000)
