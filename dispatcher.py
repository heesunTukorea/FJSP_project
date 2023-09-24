from Job import *

class Dispatcher:

    def __init__(self):
        print("dispatcher on")
        self.time = 0
    def dispatching_rule_decision(self, candidate_list, a, curr_time):
        self.time = curr_time
        if a == "random":
            coin = random.randint(0, 1)
        else:
            coin = int(a)
        if coin == 0:
            candidate_list = self.dispatching_rule_SPT(candidate_list)
            rule_name = "SPT"
        elif coin == 1:
            candidate_list = self.dispatching_rule_SSU(candidate_list)
            rule_name = "SSU"
        elif coin == 2:
            candidate_list = self.dispatching_rule_SPTSSU(candidate_list)
            rule_name = "SPTSSU"
        elif coin == 3:
            candidate_list = self.dispatching_rule_MOR(candidate_list)
            rule_name = "MOR"
        elif coin == 4:
            candidate_list = self.dispatching_rule_LOR(candidate_list)
            rule_name = "LOR"
        elif coin == 5:
            candidate_list = self.dispatching_rule_EDD(candidate_list)
            rule_name = "EDD"
        elif coin == 6:
            candidate_list = self.dispatching_rule_MST(candidate_list)
            rule_name = "MST"
        elif coin == 7:
            candidate_list = self.dispatching_rule_FIFO(candidate_list)
            rule_name = "FIFO"
        elif coin == 8:
            candidate_list = self.dispatching_rule_LIFO(candidate_list)
            rule_name = "LIFO"
        elif coin == 9:
            candidate_list = self.dispatching_rule_CR(candidate_list)
            rule_name = "CR"
        elif coin == 10:
            candidate_list = self.dispatching_rule_NONE(candidate_list)
            rule_name = "NONE"

        return candidate_list, rule_name

    def dispatching_rule_SPT(self, candidate_list):
        #candidate_list = [job, processing_time, setup_time, finish time]
        candidate_list.sort(key=lambda x: x[1], reverse=False)
        #return job, processing_time, setup_time, finish_time
        return candidate_list

    def dispatching_rule_SSU(self, candidate_list):

        candidate_list.sort(key=lambda x: x[2], reverse=False)

        return candidate_list

    def dispatching_rule_SPTSSU(self, candidate_list):
        candidate_list.sort(key=lambda x: x[1]+x[2], reverse=False)

        return candidate_list

    def dispatching_rule_MOR(self, candidate_list):
        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=True)

        return candidate_list

    def dispatching_rule_LOR(self, candidate_list):

        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=False)
        return candidate_list

    def dispatching_rule_EDD(self, candidate_list):
        candidate_list.sort(key=lambda x: x[0].duedate, reverse=False)

        return candidate_list

    def dispatching_rule_MST(self, candidate_list):
        candidate_list.sort(key=lambda x: x[0].duedate - self.time - x[1], reverse=False)

        return candidate_list

    def dispatching_rule_CR(self, candidate_list):
        candidate_list.sort(key=lambda x: (x[0].duedate - self.time) / x[1], reverse=False)
        return candidate_list

    def dispatching_rule_FIFO(self, candidate_list):

        candidate_list.sort(key=lambda x: x[0].job_arrival_time, reverse=False)
        return candidate_list

    def dispatching_rule_LIFO(self, candidate_list):
        candidate_list.sort(key=lambda x: x[0].job_arrival_time, reverse=True)
        return candidate_list

    def dispatching_rule_NONE(self, candidate_list):
        candidate_list = []
        return candidate_list
