
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import copy
import random
from matplotlib import pyplot as plt

from Parameter import *

class GanttChart:
    def __init__(self, plotlydf, plotlydf_arrival_and_due,params):
        print("gantt_on")
        self.plotlydf = plotlydf
        self.plotlydf_arrival_and_due = plotlydf_arrival_and_due
        param = params
        self.gantt_on = param.gantt_on

    def update_df(self):

        step_rule = []
        for i in range(len(self.plotlydf)):
            if str(self.plotlydf["Rule"].loc[i]) != "None":
                step_rule.append(str(self.plotlydf["Step"].loc[i]) + "-" + str(self.plotlydf["Rule"].loc[i]))
            else:
                step_rule.append("NONE")
        self.plotlydf["Step-Rule"] = step_rule

        id_op = []
        for i in range(len(self.plotlydf)):
            if str(self.plotlydf["Task"].loc[i]) != "None":
                id_op.append(str(self.plotlydf["JOB_ID"].loc[i]) + "-" + str(self.plotlydf["Task"].loc[i]))
            else:
                id_op.append("NONE")
        self.plotlydf["ID_OP"] = id_op

    
    def play_gantt(self):
        fig,fig2,fig3,fig4,fig5,fig6,fig8 = None, None, None, None, None, None, None  # 변수를 미리 None으로 초기화
        self.update_df()
        for i in self.gantt_on:
            if self.gantt_on[i]:
                if i == "main_gantt":
                    fig6 =self.main_gantt()
                elif i == "machine_on_job_number":
                    fig = self.mahicne_on_job_number()
                elif i == "machine_gantt":
                    fig2 = self.machine_gantt()
                elif i == "DSP_gantt":
                    fig3 = self.DSP_gantt()
                elif i == "step_DSP_gantt":
                    fig4 = self.step_DSP_gantt()
                elif i == "heatMap_gantt":
                    fig5 = self.heatMap_gantt()
                elif i == "job_gantt_for_Q_time":
                    fig8 = self.job_gantt_for_Q_time()
        return fig,fig2,fig3,fig4,fig5,fig6,fig8


    def mahicne_on_job_number(self):
        fig = px.bar(self.plotlydf, x="Resource", y="Type", color="Type", facet_row="Type")
        fig.update_yaxes(matches=None)
        # fig.show()
        
        [(self.modify_width(bar, 0.7))
        for bar in fig.data if ('setup' in bar.legendgroup)]
        return fig

    def machine_gantt(self):

    # fig,write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
        plotlydf3 = self.plotlydf.sort_values(by=['Type'], ascending=True)
        fig2 = px.timeline(plotlydf3, x_start="Start", x_end="Finish", y="Type", template="seaborn", color="Resource",
                           text="Resource", width=2000, height=1000)
        fig2.update_traces(marker=dict(line_color="yellow", cmid=1000))
        # fig2.show()
        return fig2
    def DSP_gantt(self):

        fig3 = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", template="simple_white", color="Type",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar)) for bar in fig3.data if ('setup' in bar.legendgroup)]
        # fig3.show()
        return fig3
    def step_DSP_gantt(self):

        fig4 = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", template="simple_white", color="Type",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Step-Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
         for bar in fig4.data if ('setup' in bar.legendgroup)]
        # fig4.show()
        return fig4
    def heatMap_gantt(self):

        fig5 = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Rule", template="simple_white", color="Rule",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Step-Rule", width=2000, height=800)
        # fig5.show()
        return fig5
    def main_gantt(self):

        fig6 = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", template="simple_white", color="Type",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="ID_OP", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
         for bar in fig6.data if ('setup' in bar.legendgroup)]

        # fig6.show()
        return fig6
    def job_gantt_for_Q_time(self):
        df = self.plotlydf_arrival_and_due._append(self.plotlydf, ignore_index=True)
        df = df.sort_values(by=['Start', "Finish"], ascending=[False, False])
        df = self.to_top_arrival_df(df)
        df = self.to_bottom_due_df(df)
        fig8 = px.timeline(df, x_start="Start", x_end="Finish", y="JOB_ID", template="simple_white", color="Q_check",
                           color_discrete_sequence=px.colors.qualitative.Dark24, text="Q_diff", width=2000, height=2000)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
         for bar in fig8.data if ('setup' in bar.legendgroup)]
        # fig8.show()
        return fig8
    def modify_width(self, bar, width):
        """
        막대의 너비를 설정합니다.
        width = (단위 px)
        """
        bar.width = width

    def modify_text(self, bar):
        """
        막대의 텍스트를 설정합니다.
        width = (단위 px)
        """
        bar.text = "su"

    def to_top_arrival_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        arrival_df = df.loc[df['Type'] == 'job_arrival']
        df = df[df['Type'] != 'job_arrival']
        arrival_df = arrival_df._append(df, ignore_index=True)
        return arrival_df

    def to_bottom_setup_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        setup_df = df.loc[df['Type'] == 'setup']
        df = df[df['Type'] != 'setup']
        df = df._append(setup_df, ignore_index=True)
        return df

    def to_bottom_due_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        setup_df = df.loc[df['Type'] == 'due_date']
        df = df[df['Type'] != 'due_date']
        df = df._append(setup_df, ignore_index=True)
        return df
    