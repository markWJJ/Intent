#!/usr/bin/python3
# coding: utf-8

import requests
import json
import datetime

ES_HOST = '192.168.3.105'  # 公司内网地址
# ES_HOST = '52.80.177.148'  # 外网30G内存对应的数据库
# ES_HOST = '52.80.187.77'  # 外网，南迪数据管理关联的es数据库；
# ES_HOST = '10.13.70.57'  # 外网词向量测试服
# ES_HOST = '10.13.70.173'  # 外网词向量正式服

ES_PORT = '9200'
# ES_PORT = '18200'
ES_INDEX = 'intent'  # 必须是小写
ES_USER = 'elastic'
ES_PASSWORD = 'webot2008'

intent_entity_dict = {
    '保障项目的责任免除详情': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '保障项目如何申请理赔': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '询问犹豫期': [['Baoxianchanpin'], []],
    '询问保险责任': [['Baoxianchanpin'], []],
    '询问合同恢复': [['Baoxianchanpin'], []],
    '询问保单借款': [['Baoxianchanpin'], []],
    '询问特别约定': [['Baoxianchanpin'], []],
    '询问合同终止': [['Baoxianchanpin'], []],
    '询问合同生效': [['Baoxianchanpin'], []],
    '询问宣告死亡': [['Baoxianchanpin'], []],
    '询问如实告知': [['Baoxianchanpin'], []],
    '询问合同解除': [['Baoxianchanpin'], []],
    '询问保险期间': [['Baoxianchanpin'], []],
    '询问保费垫缴': [['Baoxianchanpin'], []],
    '询问受益人': [['Baoxianchanpin'], []],
    '询问体检或验尸': [['Baoxianchanpin'], []],
    '询问未归还款项偿还': [['Baoxianchanpin'], []],
    '询问合同构成': [['Baoxianchanpin'], []],
    '询问争议处理': [['Baoxianchanpin'], []],
    '询问投保年龄': [['Baoxianchanpin'], []],
    '询问适用币种': [['Baoxianchanpin'], []],
    '询问基本保险金额': [['Baoxianchanpin'], []],
    '询问合同种类': [['Baoxianchanpin'], []],
    '询问等待期': [['Baoxianchanpin'], []],
    '询问减额缴清': [['Baoxianchanpin'], []],
    '询问缴费方式': [['Baoxianchanpin'], ['Jiaofeifangshi']],
    '询问缴费年期': [['Baoxianchanpin'], []],
    '服务内容介绍': [['Fuwuxiangmu'], []],
    '产品保哪些疾病': [['Baoxianchanpin'], []],
    '询问公司有哪些产品': [['Baozhangxiangmu', 'Baoxianchanpin'], []],
    '询问宽限期时间': [['Baoxianchanpin'], []],
    '询问宽限期定义': [['Baoxianchanpin'], []],
    '询问产品优势': [['Baoxianchanpin'], []],
    '询问免赔额定义': [['Baoxianchanpin'], []],
    '询问免赔额数值': [['Baoxianchanpin'], []],
    '询问保险费缴纳': [['Baoxianchanpin'], []],
    '减额缴清': [['Baoxianchanpin'], []],
    '询问诉讼时效': [['Baoxianchanpin'], []],
    '询问变更通讯方式': [['Baoxianchanpin'], []],
    '询问保险金给付': [['Baoxianchanpin'], []],
    '询问区别': [[], []],
    '咨询时间': [[], []],
    '患某疾病是否可以投保': [['Baoxianchanpin', 'Jibing'], []],
    '保单借款的还款期限': [['Baoxianchanpin'], []],
    '某种情景是否可以投保': [['Baoxianchanpin', 'Qingjing'], []],
    '保险种类的定义': [['Baoxianzhonglei'], []],
    '某情景是否在承保范围内': [['Baoxianchanpin', 'Qingjing'], []],
    '疾病种类的定义': [['Jibingzhonglei','Baoxianchanpin'], []],
    '保障项目的定义': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '得了某类疾病怎样申请理赔': [['Baoxianchanpin', 'Jibingzhonglei'], []],
    '询问失踪处理': [['Baoxianchanpin'], []],
    '疾病种类包含哪些疾病': [['Baoxianchanpin', 'Jibingzhonglei'], []],
    '某疾病是否属于某疾病种类': [['Baoxianchanpin','Jibing', 'Jibingzhonglei'], []],
    '询问信息误告': [['Baoxianchanpin'], []],
    '询问疾病种类包含哪些': [['Jibingzhonglei'], []],
    '某种疾病的预防': [['Jibing'], []],
    '某种疾病的高发区域': [['Jibing'], []],
    '某种体检异常指标分析': [[], []],
    '某种体检异常指标定义': [[], []],
    '身体器官的构成': [[], []],
    '疾病发病原因': [['Jibing'], []],
    '某项体检的包含项': [[], []],
    '某平台使用方法': [[], []],
    '询问合同领取方式': [[], ['Baoxianchanpin']],
    '投保途径': [[], ['Baoxianchanpin']],
    '投保人和被保险人关系': [['Baoxianchanpin'], []],
    '区分疾病种类的标准': [['Baoxianchanpin'], []],
    '保障项目的年龄限制': [['Baoxianchanpin','Baozhangxiangmu'], []],
    '公司经营状况': [['Baoxianchanpin'], []],
    '投保的职业要求': [['Baoxianchanpin'], []],
    '购买某项保险产品有无优惠': [['Baoxianchanpin'], []],
    '某类险种的缴费方式': [['Baoxianzhonglei'], ['Baoxianchanpin']],
    '体检的时限要求': [[], []],
    '某渠道的投保流程': [['Baoxianchanpin'], []],
    '保险费定价': [['Baoxianchanpin'], []],
    '理赔渠道申请流程': [[], []],
    '某产品的最低保额': [['Baoxianchanpin'], []],
    '理赔后缓缴保险费的处理办法': [['Baoxianchanpin'], []],
    '申请某保障项目所需资料': [[], ['Baozhangxiangmu']],
    '公司股东构成': [[], []],
    '不同保障项目的重复理赔': [['Baozhangxiangmu','Baoxianchanpin'], []],
    '询问产品价格': [['Baoxianchanpin'], []],
    '询问公司介绍': [[], []],
    '产品是否属于某一类险': [['Baoxianchanpin', 'Baoxianzhonglei'], []],
    '承保区域': [['Baoxianchanpin'], []],
    '保障项目的赔付次数': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '保费支付渠道': [['Baoxianchanpin'], []],
    '财务问卷针对的对象': [[], []],
    '疾病种类治疗费用': [['Jibingzhonglei'], []],
    '特殊人群投保规则': [['Baoxianchanpin'], []],
    '推荐保额': [['Baoxianchanpin'], []],
    '核保不通过解决办法': [[], []],
    '体检标准': [[], []],
    '投保所需资料': [['Baoxianchanpin'], []],
    '询问保险条款': [['Baoxianchanpin'], []],
    '理赔医院': [['Baoxianchanpin'], ['Yiyuan']],
    '保障项目的赔付': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '险种介绍': [[], ['Baoxianzhonglei']],
    '产品推荐': [['Baoxianchanpin'], []],
    '工具的使用限制': [[], []],
    '保险类型判断': [['Baoxianchanpin'], []],
    '诊断报告的领取': [[], []],
    '某人购买产品是否有优惠': [[], ['Baoxianchanpin']],
    '产品升级介绍': [['Baoxianchanpin'], []],
    '体检指引': [['Baoxianchanpin'], []],
    # '产品保额限制': [['Baoxianchanpin'], []],
    '某情景是否具备购买产品资格': [['Qingjing'], ['Baoxianchanpin']],
    '询问疾病种类的生存期': [['Jibingzhonglei','Baoxianchanpin'], []],
    '理赔的地域限制': [[], []],
    '特殊人群投保限额': [['Baoxianchanpin'], []],
    '权益间的关系': [['Baoxianchanpin'], []],
    '保障项目的保障范围': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '产品介绍': [['Baoxianchanpin'], []],
    '申请纸质合同': [[], []],
    '疾病种类的赔付流程': [['Baoxianchanpin', 'Baoxianzhonglei'], []],
    '理赔速度': [['Baoxianchanpin'], []],
    '产品保障的地域限制': [['Baoxianchanpin'], []],
    '合同丢失': [[], []],
    '电子合同介绍': [[], []],
    '保单查询': [[], []],
    '理赔条件': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '赔付方式': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '投保指引': [['Baoxianchanpin'], []],
    '业务办理提交材料': [[], []],
    '投保人与被保险人关系': [[], []],
    '投保人条件限制': [[], []],
    '红利申请规定': [[], []],
    '合同自动终止的情景': [[], ['Qingjing']],
    '理赔报案的目的': [[], []],
    '险种关联选择申请资料': [[], []],
    '变更投保人': [['Baoxianchanpin'], []],
    '查询方式': [[], []],
    '得了某个疾病怎样申请理赔': [['Baoxianchanpin', 'Jibing'], []],
    '疾病是否赔某个保障项目': [['Baoxianchanpin', 'Jibing', 'Baozhangxiangmu'], []],
    '情景是否赔偿某个保障项目': [['Baoxianchanpin', 'Qingjing', 'Baozhangxiangmu'], []],
    '情景的定义': [['Qingjing'], []],
    '疾病的定义': [['Jibing','Baoxianchanpin'], []],
    '询问保险事故通知': [['Baoxianchanpin'], []],
    '疾病包含哪些疾病': [['Jibing'], []],
    '如何申请某项服务': [['Fuwuxiangmu'], []],
    '释义项的定义': [['Shiyi'], []],
    '询问分公司联系方式': [[], []],
    '询问复效期': [['Baoxianchanpin'], []],
    '询问赔付比例': [['Baoxianchanpin', 'Baozhangxiangmu'], []],
    '询问投保规则': [['Baoxianchanpin'], []],
    '询问投保人': [['Baoxianchanpin'], []],
    '询问变更缴费方式': [['Baoxianchanpin'], []],
    '询问现金价值': [['Baoxianchanpin'], []],
    '询问体检标准': [[], []],
    '询问保险种类': [['Baoxianchanpin'], ['Baoxianzhonglei']],
    '保险产品的定义': [[], ['Baoxianzhonglei']],
    '某种疾病是否在承保范围内': [['Baoxianchanpin', 'Jibing'], []],
    '复效期有无收取滞纳金': [['Baoxianchanpin'], []],
    '减额缴清的影响': [['Baoxianchanpin'], []],
    '险种的年龄限制': [['Baoxianzhonglei'], []],
    # '某种疾病的保障范围': [['Jibing','Baoxianchanpin'], []],
    '高发疾病': [['Baoxianchanpin'], []],
    # '产品保障期限': [['Baoxianchanpin'], []],
    '忘记交保费的解决办法': [[], []],
    '保险是否有购买限制': [['Baoxianchanpin'], []],
    '理赔申请后出现特殊情况的解决办法': [[], ['Baoxianchanpin']],
    '豁免保费后保单是否存在现金价值': [['Baoxianchanpin'], []],
    '投保渠道的月缴首期': [[], []],
    '保障项目给付一半后的处理方法': [['Baozhangxiangmu'], []],
    '特殊人群的理赔方式': [['Baoxianchanpin'], []],
    '国外医院的要求': [[], []],
    # '保障项目的理赔方式': [['Baozhangxiangmu'], ['Baoxianchanpin']],
    '投资连结保险投资账户介绍': [[], []],
    '询问包含产品': [[], ['Baoxianchanpin']],
    '询问某产品增值服务项目': [['Baoxianchanpin', 'Fuwuxiangmu'], []],
    '询问增值服务内容': [['Baoxianchanpin', 'Fuwuxiangmu'], []],
    '询问合作医院': [[], []],
    '询问保险金额':[['Baoxianchanpin'],[]],
    # '询问保额规定': [['Baoxianchanpin'], []],
    '询问公司定义': [[], []],
    '询问险种免赔额数值': [['Baoxianzhonglei'], []],
    '询问附加险': [['Baoxianchanpin'], []],
    '询问专科医生': [[], []],
    '询问保额累计': [['Baoxianchanpin'], []],
    '询问投保书填写': [[], []],
    '询问哪个保险保疾病': [['Baoxianchanpin'], []],
    '询问合同变更': [['Baoxianchanpin'], []],
    '咨询引导': [[], []],
    '保费支出': [[], []],
    '公司地址': [[], []],
    '询问联系方式': [[], []],
    '询问合作银行': [[], []],
    '商业保险与医保的关系': [['Baoxianchanpin'], []],
    '保单回溯': [[], ['保单回溯']],
    '某体检的意义': [[], []],
    '中信银行': [[], []],
    '询问保险产品': [[], []],
    '保险产品的对比': [['Baoxianchanpin'], []],
    '疾病治疗费用': [[], []],
    '增值服务使用次数':[[],[]],
    '增值服务使用时间':[['Baoxianchanpin'],[]],
    '增值服务亮点':[[],[]],
    '变更通讯资料相关规定':[[],[]],
    '银行转账授权相关规定':[[],[]],
    '变更投保人相关规定':[[],[]],
    '更改个人身份资料相关规定':[[],[]],
    '变更签名相关规定':[[],[]],
    '保费逾期未付选择相关规定':[[],[]],
    '变更受益人相关规定':[[],[]],
    '领取现金红利相关规定':[[],[]],
    '身故保险金分期领取选择相关规定':[[],[]],
    '变更红利领取方式相关规定':[[],[]],
    '指定第二投保人相关规定':[[],[]],
    '复效相关规定':[[],[]],
    '变更职业等级相关规定':[[],[]],
    '变更缴费方式相关规定':[[],[]],
    '结束保险费缓缴期相关规定':[[],[]],
    '降低主险保额相关规定':[[],[]],
    '变更附加险相关规定':[[],[]],
    '减额缴清相关规定':[[],[]],
    '补充告知相关规定':[[],[]],
    '取消承保条件相关规定':[[],[]],
    '保单借款相关规定':[[],[]],
    '保单还款相关规定':[[],[]],
    '生存给付确认相关规定':[[],[]],
    '变更年金领取方式相关规定':[[],[]],
    '变更生存保险金领取方式相关规定':[[],[]],
    '领取生存保险金相关规定':[[],[]],
    '变更给付账号相关规定':[[],[]],
    '满期给付生存确认相关规定':[[],[]],
    '险种关联选择相关规定':[[],[]],
    '部分提取相关规定':[[],[]],
    '额外投资相关规定':[[],[]],
    '投资账户选择相关规定':[[],[]],
    '投资账户转换相关规定':[[],[]],
    '终止保险合同相关规定':[[],[]],
    '申请定期额外投资相关规定':[[],[]],
    '变更定期额外投资相关规定':[[],[]],
    '终止定期额外投资相关规定':[[],[]],
    '犹豫期减保相关规定':[[],[]],
    '犹豫期终止合同相关规定':[[],[]],
    '犹豫期其他保全变更相关规定':[[],[]],
    '补发保单相关规定':[[],[]],
    '变更通讯资料申请时间':[[],[]],
    '银行转账授权申请时间':[[],[]],
    '变更投保人申请时间':[[],[]],
    '更改个人身份资料申请时间':[[],[]],
    '变更签名申请时间':[[],[]],
    '保费逾期未付选择申请时间':[[],[]],
    '变更受益人申请时间':[[],[]],
    '领取现金红利申请时间':[[],[]],
    '身故保险金分期领取选择申请时间':[[],[]],
    '变更红利领取方式申请时间':[[],[]],
    '指定第二投保人申请时间':[[],[]],
    '复效申请时间':[[],[]],
    '变更职业等级申请时间':[[],[]],
    '变更缴费方式申请时间':[[],[]],
    '结束保险费缓缴期申请时间':[[],[]],
    '降低主险保额申请时间':[[],[]],
    '变更附加险申请时间':[[],[]],
    '减额缴清申请时间':[[],[]],
    '补充告知申请时间':[[],[]],
    '取消承保条件申请时间':[[],[]],
    '保单借款申请时间':[[],[]],
    '保单还款申请时间':[[],[]],
    '生存给付确认申请时间':[[],[]],
    '变更年金领取方式申请时间':[[],[]],
    '变更生存保险金领取方式申请时间':[[],[]],
    '领取生存保险金申请时间':[[],[]],
    '变更给付账号申请时间':[[],[]],
    '满期给付生存确认申请时间':[[],[]],
    '险种关联选择申请时间':[[],[]],
    '部分提取申请时间':[[],[]],
    '额外投资申请时间':[[],[]],
    '投资账户选择申请时间':[[],[]],
    '投资账户转换申请时间':[[],[]],
    '终止保险合同申请时间':[[],[]],
    '申请定期额外投资申请时间':[[],[]],
    '变更定期额外投资申请时间':[[],[]],
    '终止定期额外投资申请时间':[[],[]],
    '犹豫期减保申请时间':[[],[]],
    '犹豫期终止合同申请时间':[[],[]],
    '犹豫期其他保全变更申请时间':[[],[]],
    '补发保单申请时间':[[],[]],
    '保险产品支持保全项':[['Baoxianchanpin'],[]],
    '保险产品变更通讯资料的范围':[['Baoxianchanpin'],[]],
    '保险产品银行转账授权的范围':[['Baoxianchanpin'],[]],
    '保险产品更改个人身份资料的范围':[['Baoxianchanpin'],[]],
    '保险产品变更签名的范围':[['Baoxianchanpin'],[]],
    '保险产品变更受益人的范围':[['Baoxianchanpin'],[]],
    '保险产品变更投保人的范围':[['Baoxianchanpin'],[]],
    '保险产品保费逾期未付选择的范围':[['Baoxianchanpin'],[]],
    '保险产品补发保单的范围':[['Baoxianchanpin'],[]],
    '保险产品变更职业等级的范围':[['Baoxianchanpin'],[]],
    '保险产品变更缴费方式的范围':[['Baoxianchanpin'],[]],
    '保险产品变更保险计划的范围':[['Baoxianchanpin'],[]],
    '保险产品复效的范围':[['Baoxianchanpin'],[]],
    '保险产品减额缴清的范围':[['Baoxianchanpin'],[]],
    '保险产品取消承保条件的范围':[['Baoxianchanpin'],[]],
    '保险产品补充告知的范围':[['Baoxianchanpin'],[]],
    '保险产品保单借款的范围':[['Baoxianchanpin'],[]],
    '保险产品保单还款的范围':[['Baoxianchanpin'],[]],
    '保险产品终止保险合同的范围':[['Baoxianchanpin'],[]],
    '保险产品附约变更的范围':[['Baoxianchanpin'],[]],
    '保险产品满期生存确认的范围':[['Baoxianchanpin'],[]],
    '保险产品降低主险保额的范围':[['Baoxianchanpin'],[]],
    '产品续保':[['Baoxianchanpin'],[]],
    '产品期满返还':[['Baoxianchanpin'],[]],
    '产品费用与报销':[['Baoxianchanpin'],[]],
    '变更保险金额':[['Baoxianchanpin'],[]],
    '某情景下全残或身故赔偿':[['Qingjing','Baoxianchanpin'],[]],
    '保障项目保哪些疾病':[['Baozhangxiangmu'],[]],
    '询问免责条款':[['Baoxianchanpin'],[]],
    '可以豁免':[['Baoxianchanpin','Jibing'],[]],
    '理赔需要的材料':[['Baoxianchanpin'],[]],
    '保费核算':[['Baoxianchanpin'],[]],
    '询问保全变更':[[],[]],
    '不能豁免': [['Baoxianchanpin', 'Jibing'], []],
    '多次赔付':[['Baoxianchanpin', 'Jibing'],[]]

}

def download_template_intent(es_host=ES_HOST, es_port=ES_PORT, _index='templates_question', es_user=ES_USER, es_password=ES_PASSWORD, pid='all_baoxian'):
    """
    模板里头有必须实体，可选实体，以模板数据为准
    :param es_host: 
    :param es_port: 
    :param _index: 
    :param _type:

    :param es_user: 
    :param es_password: 
    :param data: 
    :param pid: 
    :return: 
    """
    es_index_alias = "{}_{}_alias".format(pid.lower(), _index)
    intent_bixuan_kexuan_dict = {}
    intent_list = list(intent_entity_dict.keys())
    try:
        # 获取全部的模板数据
        url = 'http://{}:{}/{}/_search?scroll=10m&size=5000'.format(es_host, es_port, es_index_alias)

        args_json = {
            "query" : {
                "match_all" : {}
            }
        }
        r = requests.get(url, json=args_json, auth=(es_user, es_password))
        ret = r.json()
        hits = ret['hits']['hits']
        datas = [h['_source'] for h in hits]

    except Exception as e:
        print("在索引`{}:{}/{}`下获取意图为`{}`的必须、可选参数出错： \n{}".format(ES_HOST, ES_PORT, es_index_alias, intent_list, e))
        datas = []

    for data in datas:
        intent = data.get('intent')
        pass_intent_list = ['询问增值服务种类','得了某类疾病怎样申请理赔','是否承保某个疾病','询问保额规定']
        assert intent in intent_list or intent in pass_intent_list, "模板中的意图`{}`应该在意图字典中存在".format(intent)
        bixuan = data.get('必选实体', [])
        kexuan = data.get('可选实体', [])
        if intent and (bixuan or kexuan):
            intent_bixuan_kexuan_dict.setdefault(intent, (bixuan, kexuan))

    return intent_bixuan_kexuan_dict


def intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER, es_password=ES_PASSWORD, data=None, pid='all_baoxian'):

    # 模板中存在的意图集合
    templates_intent_bixuan_kexuan_dict = download_template_intent(es_host=es_host, es_port=es_port,
                                                                   _index='templates_question', es_user=es_user,
                                                                   es_password=es_password, pid=pid)

    es_index = "{}_{}".format(pid, _index)
    now = datetime.datetime.now()
    index_end = now.strftime('%Y%m%d_%H%M%S')
    current_es_index = "{}_{}".format(es_index, index_end).lower()

    alias_name = '{}_alias'.format(es_index)

    url = "http://{}:{}/{}/{}/_bulk".format(es_host, es_port, current_es_index, _type)

    all_data = ''
    # del_alias_name = {"delete": {"_index": alias_name}}
    # all_data += json.dumps(del_alias_name, ensure_ascii=False) + '\n'
    for template_id, (intent, entity_list) in enumerate(data.items()):
        # 若意图在模板库中存在，则以模板的意图为准
        if templates_intent_bixuan_kexuan_dict.get(intent):
            bixuan, kexuan = templates_intent_bixuan_kexuan_dict.get(intent)
        else:
            bixuan, kexuan = entity_list
        doc = {"intent": intent, "必选实体": bixuan, "可选实体": kexuan, '模板id': template_id}
        create_data = {"create": {"_id": template_id}}
        all_data += json.dumps(create_data, ensure_ascii=False) + '\n'
        all_data += json.dumps(doc, ensure_ascii=False) + '\n'

    ret = requests.post(url=url, data=all_data.encode('utf8'), auth=(es_user, es_password))
    # print(ret.json())

    # 添加别名
    data = {
        "actions": [
            {"remove": {
                "alias": alias_name,
                "index": "_all"
            }},
            {"add": {
                "alias": alias_name,
                "index": current_es_index
            }}
        ]
    }
    url = "http://{}:{}/_aliases".format(es_host, es_port)
    r = requests.post(url, json=data, auth=(es_user, es_password))
    print(r.json())

def create_one(_id, intent, bixuan, kexuan, es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', pid='all_baoxian'):
    """
    向意图表中插入单条数据
    :param _id: 
    :param intent: 
    :param bixuan: 
    :param kexuan: 
    :return: 
    """
    es_index = "{}_{}".format(pid, _index)
    alias_name = '{}_alias'.format(es_index)

    url = "http://{}:{}/{}/{}/{}".format(es_host, es_port, alias_name, _type, _id)
    template_id = _id
    doc = {"intent": intent, "必选实体": bixuan, "可选实体": kexuan, '模板id': template_id}
    r = requests.post(url, json=doc, auth=(ES_USER, ES_PASSWORD))
    print(r.json())

def main():
    # es_user = 'elastic'
    # es_password = 'webot2008'
    # es_host = '192.168.3.145'
    # es_port = '9200'
    # _index = 'intent'
    # _type = 'intent'

    intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER,
                 es_password=ES_PASSWORD, data=intent_entity_dict, pid='all_baoxian')


    # 插入单条：
    # _id = 144
    # intent = '测试意图'
    # bixuan = ['Jibing']
    # kexuan = []
    # create_one(_id, intent, bixuan, kexuan, _index=ES_INDEX, _type='intent', pid='all_baoxian')

if __name__ == '__main__':
    main()
