"""
Created in Wed Aug 22 2018
Revised lastly on Thurs Sep 14 2018

@author: li yifan
"""
from __future__ import print_function, absolute_import, unicode_literals, division
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import os
import shutil
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer,Image,Table,TableStyle, PageBreak
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
from operator_lib import *
"""
import math
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
"""
'''
本代码适用于数字货币的单因子测试
测试前，将本代码，包含相应因子数据的test_factor文件夹和包含收盘价、因子方向、货币交易spread数据的test_price文件夹置于同一目录下
执行时，会生成result文件夹，其中包含个因子的单独测试数据以及汇总数据的pdf和csv
在执行时，程序会判断并删除路径下的已有的同名result文件夹，请保证result文件夹和其中的文件没有被打开，若须保存之前的回测数据或同名文件夹中的其他数据，请在执行本程序前将文件夹移出该路径
'''
"""默认时间(index)为标准格式：%Y-%m-%d %H:%M:%S"""


def standard_operater(factor,date_length,direction,buy_quantile,sell_quantile,**kwargs):
    up_threshold = factor.expanding().quantile(axis=0, quantile=buy_quantile)
    down_threshold = factor.expanding().quantile(axis=0, quantile=sell_quantile)
    if direction == "Ascending":
        long = pd.DataFrame(1, index=factor.index, columns=factor.columns)
        long = long.where(factor > up_threshold, 0)
        short = pd.DataFrame(-1, index=factor.index, columns=factor.columns)
        short = short.where(factor < down_threshold, 0)
    else:
        long = pd.DataFrame(1, index=factor.index, columns=factor.columns)
        long = long.where(factor < down_threshold, 0)
        short = pd.DataFrame(-1, index=factor.index, columns=factor.columns)
        short = short.where(factor > up_threshold, 0)
    strategy = long + short
    return strategy


class Backtest(object):

    def __init__(self,**kwargs):
        if 'date_length' in kwargs:
            self.date_length=kwargs['date_length']
        else:
            self.date_length=0
        if 'date_length2' in kwargs:
            self.date_length2=kwargs['date_length2']
        else:
            self.date_length2=0
        if 'cost' in kwargs:
            self.cost=kwargs['cost']
        else:
            self.cost=0.001
        if os.path.exists('./result'):
            shutil.rmtree(os.path.join(os.getcwd(), 'result'))
        os.mkdir('result')
        '''读取所需数据'''
        self.price_filename_path = os.path.join(os.getcwd(), 'test_price')
        self.closePrice = pd.read_csv(os.path.join(self.price_filename_path, 'closePrice.csv'), index_col=0)
        self.spread_list = pd.read_csv(os.path.join(self.price_filename_path, 'spread.csv'), index_col=0)
        self.direction_list = pd.read_csv(os.path.join(self.price_filename_path, 'direction.csv'), index_col=0)
        if 'start' in kwargs:
            if kwargs['start']<self.closePrice.index.values[0]:
                self.start=self.closePrice.index.values[0]
            else:
                self.start=kwargs['start']
        else:
            self.start=self.closePrice.index.values[0]
        if 'end' in kwargs:
            if kwargs['end']>self.closePrice.index.values[-1]:
                self.end=self.closePrice.index.values[-1]
            else:
                self.end=kwargs['start']
        else:
            self.end=self.closePrice.index.values[-1]
        if 'operator' in kwargs:
            self.operator=kwargs['operator']
        else:
            self.operator=standard_operater
        if 'buy_quantile' in kwargs:
            self.buy_quantile=kwargs['buy_quantile']
        else:
            self.buy_quantile=2/3
        if 'sell_quantile' in kwargs:
            self.sell_quantile=kwargs['sell_quantile']
        else:
            self.sell_quantile=1/3
        self.factor_path = os.path.join(os.getcwd(), 'test_factor')
        self.factors_filename = os.listdir(self.factor_path)
        self.factors_name = list(map(lambda x: x[:-4], self.factors_filename))
        self.result_gather = pd.DataFrame(columns=['factor_name', 'Annualized_return_without_fee(%)',
                                              'Annualized_volatlity_wihout_fee(%)', 'Sharpe_Ratio_without_fee',
                                              'Annualized_return_with_fee(%)', 'Annualized_volatlity_wih_fee(%)',
                                              'Sharpe_Ratio_with_fee'])
        self.story=[]
        self.datetime_time_line = list(
            map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), self.closePrice.index.values))  # 作图时作为x轴

    def picture_wealth_curve_without_fee(self,diff_for_spread,result_path):
        diff_for_symbol2 = diff_for_spread.sum(axis=1) * self.wealthcurve_without_fee
        diff_for_symbol2 = pd.DataFrame(diff_for_symbol2)
        symbol_buy2 = diff_for_symbol2[diff_for_symbol2 > 0].values.flatten()
        symbol_sell2 = diff_for_symbol2[diff_for_symbol2 < 0].values.flatten()
        plt.plot(self.datetime_time_line, self.wealthcurve_without_fee, 'b')
        ymin,ymax=plt.ylim()
        delta=(ymax-ymin)/50
        plt.scatter(self.datetime_time_line, symbol_buy2 + delta, s=0.5, c='r', marker='^')
        plt.scatter(self.datetime_time_line, -symbol_sell2 - delta, s=0.5, c='g', marker='v')
        plt.ylabel('Wealth Curve without fee')
        plt.title('Wealth Curve without fee')
        plt.tick_params(labelsize=7)
        plt.legend(["Wealth Curve without fee", "Buy Symbol", "Sell Symbol"], loc='best')
        plt.savefig(os.path.join(result_path, 'Wealth Curve without fee.png'))
        # plt.show()
        plt.close()

    def picture_wealth_curve_with_fee(self,diff_for_spread,result_path):
        diff_for_symbol1 = diff_for_spread.sum(axis=1) * self.wealthcurve
        diff_for_symbol1 = pd.DataFrame(diff_for_symbol1)
        symbol_buy1 = diff_for_symbol1[diff_for_symbol1 > 0].values.flatten()
        symbol_sell1 = diff_for_symbol1[diff_for_symbol1 < 0].values.flatten()
        plt.plot(self.datetime_time_line, self.wealthcurve, 'b')
        ymin, ymax = plt.ylim()
        delta = (ymax - ymin) / 50
        plt.scatter(self.datetime_time_line, symbol_buy1 + delta, s=0.5, c='r', marker='^')
        plt.scatter(self.datetime_time_line, -symbol_sell1 - delta, s=0.5, c='g', marker='v')
        plt.ylabel('Wealth Curve with fee')
        plt.title('Wealth Curve with fee')
        plt.tick_params(labelsize=7)
        plt.legend(["Wealth Curve with fee", "Buy Symbol", "Sell Symbol"], loc='best')
        plt.savefig(os.path.join(result_path, 'Wealth Curve with fee.png'))
        # plt.show()
        plt.close()

    def picture_win(self,return_for_wealth2,result_path):
        return_for_wealth2 = pd.DataFrame(return_for_wealth2) - 1
        return_for_win_p = return_for_wealth2[return_for_wealth2 > 0].values.flatten()
        return_for_win_n = return_for_wealth2[return_for_wealth2 < 0].values.flatten()
        plt.figure()
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))  # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
        plt.ylim(-0.01, 0.01)
        plt.bar(self.datetime_time_line, return_for_win_p, color='red', width=0.005)
        plt.bar(self.datetime_time_line, return_for_win_n, color='green', width=0.005)
        plt.ylabel('Win')
        plt.title('Win')
        plt.tick_params(labelsize=7)
        plt.legend(['positive daily return', 'negative daily return'], loc='best')
        plt.axis('on')
        plt.savefig(os.path.join(result_path, 'Win.png'))
        # plt.show()
        plt.close()

    def draw_picture(self,diff_for_spread,result_path,return_for_wealth2):
        Backtest.picture_wealth_curve_with_fee(self,diff_for_spread=diff_for_spread,result_path=result_path)
        Backtest.picture_wealth_curve_without_fee(self,diff_for_spread=diff_for_spread,result_path=result_path)
        Backtest.picture_win(self,return_for_wealth2=return_for_wealth2,result_path=result_path)

    def write_excel(self,return_for_wealth1,return_for_wealth2,factor_name):
        Annualized_return_without_fee = (self.wealthcurve_without_fee.values[-1]) ** (
                365 * 24 * 60 / self.wealthcurve_without_fee.shape[0]) - 1
        Annualized_return_with_fee = (self.wealthcurve.values[-1]) ** (
                365 * 24 * 60 / self.wealthcurve.shape[0]) - 1
        Annualized_volatlity_wihout_fee = np.std(return_for_wealth2, ddof=1) * np.sqrt(365 * 24 * 60)
        Annualized_volatlity_wih_fee = np.std(return_for_wealth1, ddof=1) * np.sqrt(365 * 24 * 60)
        Sharpe_Ratio_without_fee = Annualized_return_without_fee / Annualized_volatlity_wihout_fee
        Sharpe_Ratio_with_fee = Annualized_return_with_fee / Annualized_volatlity_wih_fee
        self.result_gather = self.result_gather.append(pd.DataFrame({'factor_name': [factor_name],
                                                                     'Annualized_return_without_fee(%)': [
                                                                         Annualized_return_without_fee * 100],
                                                                     'Annualized_volatlity_wihout_fee(%)': [
                                                                         Annualized_volatlity_wihout_fee * 100],
                                                                     'Sharpe_Ratio_without_fee': [
                                                                         Sharpe_Ratio_without_fee],
                                                                     'Annualized_return_with_fee(%)': [
                                                                         Annualized_return_with_fee * 100],
                                                                     'Annualized_volatlity_wih_fee(%)': [
                                                                         Annualized_volatlity_wih_fee * 100],
                                                                     'Sharpe_Ratio_with_fee': [Sharpe_Ratio_with_fee]}))

    def write_pdf_title(self):
        '''编写pdf文件标题'''
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet['Normal']
        curr_date = time.strftime("%Y-%m-%d", time.localtime())
        rpt_title = '<para autoLeading="off" fontSize=15 align=center><font name="Helvetica"><b>Single Factor Strategy Backtest in %s</b></font><br/><br/></para>' % curr_date
        self.story.append(Paragraph(rpt_title, normalStyle))
        str_parameter = '<para autoLeading="off" fontSize=10><font name="Helvetica">Parameters of this backtest : %s</font><br/><br/></para>' % (
                '<br/>1.cost : ' + str(
            self.cost) + '<br/>2.start time : ' + self.start + '<br/>3.end time : ' + self.end)
        self.story.append(Paragraph(str_parameter, normalStyle))

    def write_pdf_picture(self,result_path):
        img1 = Image(os.path.join(result_path, 'Wealth Curve with fee.png'))
        img1.drawHeight = 200
        img1.drawWidth = 250
        self.story.append(img1)
        img2 = Image(os.path.join(result_path, 'Wealth Curve without fee.png'))
        img2.drawHeight = 200
        img2.drawWidth = 250
        self.story.append(img2)
        img3 = Image(os.path.join(result_path, 'Win.png'))
        img3.drawHeight = 200
        img3.drawWidth = 250
        self.story.append(img3)

    def get_excel(self):
        self.result_gather = self.result_gather[['factor_name', 'Annualized_return_without_fee(%)',
                                                 'Annualized_volatlity_wihout_fee(%)', 'Sharpe_Ratio_without_fee',
                                                 'Annualized_return_with_fee(%)', 'Annualized_volatlity_wih_fee(%)',
                                                 'Sharpe_Ratio_with_fee']]
        self.result_gather.to_csv(os.path.join(os.getcwd(), 'result', 'result_gather.csv'))

    def write_pdf_table_get_pdf(self,curr_date):
        '''考虑到pdf排版，将csv结果加入pdf时，将其拆成两个部分'''
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet['Normal']
        result_gather1 = self.result_gather[['factor_name', 'Annualized_return_without_fee(%)',
                                             'Annualized_volatlity_wihout_fee(%)', 'Sharpe_Ratio_without_fee']]

        result_gather2 = self.result_gather[
            ['factor_name', 'Annualized_return_with_fee(%)', 'Annualized_volatlity_wih_fee(%)',
             'Sharpe_Ratio_with_fee']]
        table_result_gather1 = Table(
            [result_gather1.columns[:, ].values.astype(str).tolist()] + result_gather1.values.tolist(),
            colWidths=[70, 170, 170, 170])
        table_result_gather2 = Table(
            [result_gather2.columns[:, ].values.astype(str).tolist()] + result_gather2.values.tolist(),
            colWidths=[70, 170, 170, 170])
        self.story.append(Paragraph(
            '<para autoLeading="off" fontSize=12><font name="Helvetica"><br/><br/>Gather Result(without fee):</font><br/><br/></para>',
            normalStyle))
        self.story.append(table_result_gather1)
        self.story.append(Paragraph(
            '<para autoLeading="off" fontSize=12><font name="Helvetica"><br/><br/>Gather Result(with fee):</font><br/><br/></para>',
            normalStyle))
        self.story.append(table_result_gather2)
        doc = SimpleDocTemplate(os.path.join(os.getcwd(), 'result', 'Bakctest_Summary in ' + curr_date + '.pdf'))
        doc.build(self.story)

    def write_pdf_subtitle(self,count):
        stylesheet = getSampleStyleSheet()
        normalStyle = stylesheet['Normal']
        str_num = '<para autoLeading="off" fontSize=10><font name="Helvetica"> Factor %s</font><br/><br/></para>' % (
                str(count + 1) + ' of ' + str(len(self.factors_name)) + ': ' + self.factors_name[count])
        self.story.append(Paragraph(str_num, normalStyle))

    def get_factor_result(self,count,diff_for_spread,return_for_wealth1,return_for_wealth2):
        '''创建因子文件夹，保存相应数据，并将相应结果加到pdf和csv文件中'''
        result_name = "result_" + self.factors_name[count]
        result_path = os.path.join(os.getcwd(), 'result', result_name)
        os.mkdir(result_path)
        self.wealthcurve.to_csv(os.path.join(result_path, 'net_value_with_fee.csv'))
        self.wealthcurve_without_fee.to_csv(os.path.join(result_path, 'net_value_without_fee.csv'))
        self.strategy.to_csv(os.path.join(result_path, 'position.csv'))

        Backtest.draw_picture(self, diff_for_spread=diff_for_spread, return_for_wealth2=return_for_wealth2,
                              result_path=result_path)
        Backtest.write_excel(self, return_for_wealth1=return_for_wealth1, return_for_wealth2=return_for_wealth2,
                             factor_name=self.factors_name[count])
        Backtest.write_pdf_picture(self, result_path=result_path)

    def run(self):
        curr_date = time.strftime("%Y-%m-%d", time.localtime())
        Backtest.write_pdf_title(self)
        for i in range(len(self.factors_name)):
            Backtest.write_pdf_subtitle(self,i)

            '''读取并按起止时间截取因子数据'''
            factor = pd.read_csv(self.factor_path + '/' + self.factors_filename[i], index_col=0)
            factor = factor.loc[self.start:self.end]
            closePrice = self.closePrice[factor.columns.values]
            new_spread_list = self.spread_list[factor.columns.values]
            direction = self.direction_list.loc[self.factors_name[i]].values[0]
            self.strategy=self.operator(factor=factor,date_length=self.date_length,direction=direction,buy_quantile=self.buy_quantile,sell_quantile=self.sell_quantile,date_length2=self.date_length2)
            weight = 1 / len(factor.columns.values)
            '''考虑交易时由spread产生的实际价格'''
            diff_for_spread = self.strategy.diff(1)
            diff_for_spread = (diff_for_spread / np.abs(diff_for_spread)).fillna(0)
            strike_price = closePrice + diff_for_spread * new_spread_list.values[0] / 2
            '''分别生成有无交易费用时的wealth curve'''
            f_return_without_fee = ((strike_price.shift(-1).pct_change()).fillna(0)) * self.strategy + 1
            diff_for_cost = np.abs(diff_for_spread)
            f_return_with_fee = f_return_without_fee * (1 - self.cost * diff_for_cost)
            return_for_wealth1 = f_return_with_fee.sum(axis=1) * weight
            self.wealthcurve = return_for_wealth1.cumprod()
            return_for_wealth2 = f_return_without_fee.sum(axis=1) * weight
            self.wealthcurve_without_fee = return_for_wealth2.cumprod()

            Backtest.get_factor_result(self,count=i,diff_for_spread=diff_for_spread,return_for_wealth1=return_for_wealth1,return_for_wealth2=return_for_wealth2)

        Backtest.get_excel(self)
        Backtest.write_pdf_table_get_pdf(self,curr_date=curr_date)


if __name__=='__main__':
    """参数在这里设置(均为可选参数)，目前可以有：cost,start,end,date_length,operator,buy_quantile,sell_quantile"""
    backtest=Backtest()
    backtest.run()






