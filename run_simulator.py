from DQN import *
from simulator_DFJSP import *
from Parameter import *

class Run_Simulator:
    def __init__(self,params):
        print("simulator on")
        #self.params = Parameters()
        self.params = params
        self.DQN = DQN(self.params.data, self.params.r_param,self.params)
        self.simulator = FJSP_simulator(self.params.data["p_data"],self.params.data["s_data"],
                                        self.params.data["q_data"],self.params.data["rd_data"],self.params)

    def main(self, mode, dsp_rule):
        if mode == "DQN":
            fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = self.DQN.main_d()
            return fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time
        elif mode == "DSP_run":
            fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = self.simulator.run(dsp_rule)
            return fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time
        elif mode == "DSP_check_run":
            for i in self.params.DSP_rule_check:
                if self.params.DSP_rule_check[i]:
                    print(i)
                    self.simulator.reset()
                    fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time=self.simulator.run(self.params.select_DSP_rule[i])
                    return fig,fig2,fig3,fig4,fig5,fig6,fig8,Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time
# if True:
#     simulator = Run_Simulator()
#     simulator.main("DQN","SSU") # dsp_rule = 개별 확인할 때만 사용하면 됨

# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
