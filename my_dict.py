def get_train_dict():
    train_dict = {'苹果健康':0,
                  '苹果黑星病一般':1,
                  '苹果黑星病严重':2,
                  '苹果灰斑病':3,
                  '苹果雪松锈病一般':4,
                  '苹果雪松锈病严重':5,
                  '樱桃健康':6,
                  '樱桃白粉病一般':7,
                  '樱桃白粉病严重':8,
                  '玉米健康':9,
                  '玉米灰斑病一般':10,
                  '玉米灰斑病严重':11,
                  '玉米锈病一般':12,
                  '玉米锈病严重':13,
                  '玉米叶斑病一般':14,
                  '玉米叶斑病严重':15,
                  '玉米花叶病毒病':16,
                  '葡萄健康':17,
                  '葡萄黑腐病一般':18,
                  '葡萄黑腐病严重':19,
                  '葡萄轮斑病一般':20,
                  '葡萄轮斑病严重':21,
                  '葡萄褐斑病一般':22,
                  '葡萄褐斑病严重':23,
                  '柑桔健康':24,
                  '柑桔黄龙病一般':25,
                  '柑桔黄龙病严重':26,
                  '桃子健康':27,
                  '桃疮痂病一般':28,
                  '桃疮痂病严重':29,
                  '辣椒健康':30,
                  '辣椒疮痂病一般':31,
                  '辣椒疮痂病严重':32,
                  '马铃薯健康':33,
                  '马铃薯早疫病一般':34,
                  '马铃薯早疫病严重':35,
                  '马铃薯晚疫病一般':36,
                  '马铃薯晚疫病严重':37,
                  '草莓健康':38,
                  '草莓叶枯病一般':39,
                  '草莓叶枯病严重':40,
                  '番茄健康':41,
                  '番茄白粉病一般':42,
                  '番茄白粉病严重':43,
                  '番茄疮痂病一般':44,
                  '番茄疮痂病严重':45,
                  '番茄早疫病一般':46,
                  '番茄早疫病严重':47,
                  '番茄晚疫病菌一般':48,
                  '番茄晚疫病菌严重':49,
                  '番茄叶霉病一般':50,
                  '番茄叶霉病严重':51,
                  '番茄斑点病一般':52,
                  '番茄斑点病严重':53,
                  '番茄斑枯病一般':54,
                  '番茄斑枯病严重':55,
                  '番茄红蜘蛛损伤一般':56,
                  '番茄红蜘蛛损伤严重':57,
                  '番茄黄化曲叶病毒病一般':58,
                  '番茄黄化曲叶病毒病严重':59,
                  '番茄花叶病毒病':60,
                  }
    return train_dict
def get_predict_dict():
    train_dict = get_train_dict()
    predict_dict = {str(train_dict[i]):i for i in train_dict}
    return predict_dict
#print(predict_dict)
