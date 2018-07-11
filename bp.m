[class,H2S,C7H8,C6H6,NO,CO,TGS822,TGS826,TGS8669,TGS2600,TGS2602,TGS2620,MS1100,MR516,TGS2444,WSP2110,NAP55A,TEMP2,HUMD2]=textread('traindata.txt');
MEXTRIB=[class,H2S,C7H8,C6H6,NO,CO,TGS822,TGS826,TGS8669,TGS2600,TGS2602,TGS2620,MS1100,MR516,TGS2444,WSP2110,NAP55A,TEMP2,HUMD2];

%将矩阵重新进行随机抽取组成新矩阵
listB=size(MEXTRIB,1);
K = ceil(1.0*listB);
MEXTRIA=MEXTRIB(randperm(listB,K),:);

%取出矩阵mextria中的特征值与类别
FEATURE=MEXTRIA(:,(2:19));
CLASS=MEXTRIA(:,1);

% 特征值归一化
[input] = mapminmax(FEATURE');

% 构造输出矩阵
% 现在是二分类问题
s = length(CLASS);
output = zeros(s,2);
for i = 1:s
    output(i, CLASS(i)+1 ) = 1;
end

%创建BP网络
%18个特征值输入，输出为2分类
%三层网络，分别有18,10,2个神经元，激活函数为logsig线性与对数S型转移
%学习目标为traingdx　梯度下降法
net = newff(input, output', [18 10],{'logsig' 'tansig' 'tansig'}, 'traingda');

%设置训练参数
net.trainparam.show = 50;    % 显示中间结果的周期
net.trainparam.epochs = 500; % 最大学习次数，迭代次数
net.trainparam.goal = 0.01;  % 训练的目标误差
net.trainparam.lr = 0.05;    % 学习速率
% net.divideFcn = '';

%开始训练
net = train(net,input,output');




