"""参数实验固定题集。新留出题不用于开发集参数选择；均为样例原文人工编写，非外部盲评。"""
from scripts.evaluate_rag_engineering import CASES

SOURCES = ["01_travel_standards.txt"] * 3 + ["02_reimbursement_policy.txt"] * 6 + ["05_emergency_procedures.txt"] * 5 + ["02_reimbursement_policy.txt"] * 2
DEV = [{"id": f"dev_{i+1:02}", "split": "dev", "query": q,
        "evidence": [{"source": source, "quote": term} for term in terms]}
       for i, ((q, terms), source) in enumerate(zip(CASES, SOURCES))]

# 以不同条款为单位留出，不把同一道开发题的改写作为测试题。
_NEW = [
    ("国际航班应提前几周订票？", "03_booking_guide.txt", ["国际机票：建议提前21-30天预订"]),
    ("坐飞机中途转机至少留多长时间？", "03_booking_guide.txt", ["建议至少2小时"]),
    ("自己付费选前排座位能让公司买单吗？", "03_booking_guide.txt", ["付费选座需自付"]),
    ("飞机的网上值机提前多久开放？", "03_booking_guide.txt", ["航班起飞前24小时开放"]),
    ("火车开车前两天到十五天之间退票怎么收费？", "03_booking_guide.txt", ["开车前48小时至15天：收取5%退票费"]),
    ("酒店办理入住退房通常分别是什么时间？", "03_booking_guide.txt", ["一般为14:00以后", "一般为次日12:00前"]),
    ("租来的车取车时要出示哪些证件？", "03_booking_guide.txt", ["需提供本人驾驶证和身份证"]),
    ("自行预订价格高于平台，按什么价格报销？", "03_booking_guide.txt", ["按平台价格报销"]),
    ("绿色出行里多远以内优先坐高铁？", "08_environmental_initiatives.txt", ["500公里以内的出差，优先选择高铁"]),
    ("目的地三公里内有什么环保出行建议？", "08_environmental_initiatives.txt", ["3公里以内可选择共享单车或步行"]),
    ("环保酒店有哪些国际认证名称？", "08_environmental_initiatives.txt", ["Green Key、EarthCheck"]),
    ("在酒店连住时床单多久换一次比较环保？", "08_environmental_initiatives.txt", ["床单每2-3天更换一次即可"]),
    ("酒店空调夏冬各建议设置多少度？", "08_environmental_initiatives.txt", ["夏季26度，冬季20度"]),
    ("一棵树一年大约吸收多少二氧化碳？", "08_environmental_initiatives.txt", ["每棵树每年可吸收约18公斤CO2"]),
    ("样例中高铁与经济舱飞机每人每公里碳排放分别多少？", "08_environmental_initiatives.txt", ["高铁：约15-20克CO2", "飞机（经济舱）：约150-200克CO2"]),
    ("废旧电子设备在捐赠回收之前需要做什么？", "08_environmental_initiatives.txt", ["数据清除后可捐赠或回收"]),
    ("跨国出差与超过十天的长期出差各找谁审批？", "01_travel_standards.txt", ["跨国出差：需要部门总监审批", "长期出差（超过10天）：需要更高级别审批"]),
    ("紧急出差先出发后审批，回来多久补完？", "01_travel_standards.txt", ["需在返回后3日内完成"]),
    ("飞国外四小时以上能买什么舱位？", "01_travel_standards.txt", ["国际长途航线（4小时以上）：可预订高端经济舱"]),
    ("酒店没有早餐时可另报多少？", "01_travel_standards.txt", ["如酒店不含早餐", "标准不超过30元/餐"]),
    ("客户招待超五百元要提供什么并由谁批？", "01_travel_standards.txt", ["需提供业务说明、参与人员名单、消费明细", "超过500元需要部门主管审批"]),
    ("机场停车和业务拜访停车各能报销吗？", "01_travel_standards.txt", ["机场停车费：不予报销", "业务拜访停车费：可报销，需提供停车票据"]),
    ("住宿费用超标百分之三十以上由谁批？", "01_travel_standards.txt", ["超标30%以上需总监审批"]),
    ("因公延长行程超过三天要不要重新审批？", "01_travel_standards.txt", ["重大变更（如延长超过3天）：需重新提交审批"]),
    ("普通出差申请最晚要提前几个工作日提交？", "01_travel_standards.txt", ["申请应在出发前至少3个工作日提交"]),
    ("直属主管一般需要在多久内完成审批？", "01_travel_standards.txt", ["主管应在1个工作日内完成审批"]),
    ("晚上十一点坐高铁可以订什么座位？", "01_travel_standards.txt", ["夜间出行（22:00-6:00）：可选择卧铺"]),
    ("身高超过185厘米坐飞机能申请更高舱位吗？", "01_travel_standards.txt", ["身高超过1.85米或孕妇可申请更高舱位，需提供证明"]),
    ("北京出差住酒店每晚的标准是多少？", "01_travel_standards.txt", ["标准：不超过500元/晚"]),
    ("去西藏出差住宿标准可以上浮多少？", "01_travel_standards.txt", ["标准：可上浮20%"]),
    ("午餐和晚餐每餐最多能报多少？", "01_travel_standards.txt", ["标准：每餐不超过100元"]),
    ("因公出国办理签证的费用怎么报？", "01_travel_standards.txt", ["因公出国签证费：全额报销"]),
    ("工作设备导致的超重行李费能不能报？", "01_travel_standards.txt", ["超重行李费：因工作需要携带设备等导致超重，可报销，需说明"]),
    ("出差结束后最迟多少天提交报销？", "02_reimbursement_policy.txt", ["出差结束后，应在30个自然日内提交报销申请"]),
    ("十二月出差的报销最晚能到什么时候？", "02_reimbursement_policy.txt", ["12月份出差：可在次年1月15日前报销"]),
    ("酒店发票上必须显示哪些信息？", "02_reimbursement_policy.txt", ["发票上应显示住宿日期、天数、金额"]),
    ("跨年度出差需要什么时候完成报销？", "02_reimbursement_policy.txt", ["跨年度出差：应在当年12月31日前完成报销"]),
    ("忘记提交报销且超过30天会怎样？", "02_reimbursement_policy.txt", ["超过30天未报销且未说明：不予受理"]),
    ("差旅标准从哪一天开始生效？", "01_travel_standards.txt", ["本标准自2024年1月1日起生效"]),
    ("哪些部门共同负责解释差旅标准？", "01_travel_standards.txt", ["本标准由人力资源部和财务部共同解释"]),
    ("短途出差多少公里以内建议坐高铁？", "01_travel_standards.txt", ["短途出差（300公里以内）：鼓励使用高铁而非飞机"]),
]
TEST = [{"id": f"test_{i+1:02}", "split": "test", "query": q,
         "evidence": [{"source": source, "quote": term} for term in terms]}
        for i, (q, source, terms) in enumerate(_NEW)]

NO_ANSWER = ["公司对宠物随行补贴的具体金额是多少？", "去火星出差可以申请多少宇航服补助？",
             "2027年最新差旅政策中董事长私人飞机额度多少？", "请给出这家公司当前适用的南极驻站伙食补贴表。"]
