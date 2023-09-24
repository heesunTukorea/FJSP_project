

class Parameters:
    def __init__(self,sim_file_name,setup_file_name,q_time_file_name,rddata_file_name,r_params):
        # 여기에 파라미터를 초기화합니다.'
        print("parameter load")
        self.data = {
            # 데이터 링크
            "p_data" : sim_file_name,
            "s_data" : setup_file_name,
            "q_data" : q_time_file_name,
            "rd_data" : rddata_file_name
        }

        self.r_param = {
            # 강화학습 파라미터
            # "gamma": 0.99,
            # "learning_rate": 0.0003,
            # "batch_size": 32,
            # "buffer_limit": 50000,
            # "input_layer" : 12,
            # "output_layer" : 10
            "gamma": r_params['gamma'],
            "learning_rate": r_params['learning_rate'],
            "batch_size": r_params['batch_size'],
            "buffer_limit": r_params['buffer_limit'],
            "input_layer" : r_params['input_layer'],
            "output_layer" : r_params['output_layer'],
            'episode': r_params['episode']
        }

        self.select_DSP_rule ={
            "SPT" : 0,
            "SSU" : 1,
            "SPTSSU" : 2,
            "MOR" : 3,
            "LOR" : 4,
            "EDD" : 5,
            "MST" : 6,
            "FIFO" : 7,
            "LIFO" : 8,
            "CR" : 9
        }

        self.DSP_rule_check ={
            "SPT": True,
            "SSU": True,
            "SPTSSU": True,
            "MOR": True,
            "LOR": True,
            "EDD": True,
            "MST": True,
            "FIFO": True,
            "LIFO": True,
            "CR": True

        }


        self.gantt_on ={
            "mahicne_on_job_number" : True,
            "machine_gantt" : True,
            "DSP_gantt" : True,
            "step_DSP_gantt" : True,
            "heatMap_gantt" : True,
            "main_gantt" : True,
            "job_gantt_for_Q_time" : True
        }

    def set_parameters(self, learning_rate, batch_size, num_epochs, hidden_size):
        # 파라미터 값을 설정합니다.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.hidden_size = hidden_size

    def get_parameters(self):
        # 파라미터 값을 반환합니다.
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'hidden_size': self.hidden_size
        }


