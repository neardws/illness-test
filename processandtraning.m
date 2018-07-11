% 读取原始数据
original=textread('data.dat','%s'); %读取原始数据
numofframe=length(original)/46;     %46为每一帧的数据个数，numoffranme为帧数
frequency=(numofframe-1)/((datenum(original(length(original)-44),'HH:MM:SS.FFF')-datenum(original(2),'HH:MM:SS.FFF'))*24*3600);
originalreshaped=reshape(original,46,numofframe);%组成cell矩阵
delstr=originalreshaped(3:end,:);%删除cell内的字符串
todec=hex2dec(delstr);%将cell内的16进制转换成10进制
reshapeddec=reshape(todec,44,numofframe);%组成矩阵

H2S=(reshapeddec(1,:)*256+reshapeddec(2,:))*0.1;
C7H8=(reshapeddec(3,:)*256+reshapeddec(4,:))*0.1;
C6H6=(reshapeddec(5,:)*256+reshapeddec(6,:))*0.1;
NO=(reshapeddec(7,:)*256+reshapeddec(8,:))*0.01;
CO=(reshapeddec(9,:)*256+reshapeddec(10,:))*0.1;
TGS822=reshapeddec(11,:)*256+reshapeddec(12,:);
TGS826=reshapeddec(13,:)*256+reshapeddec(14,:);
TGS8669=reshapeddec(15,:)*256+reshapeddec(16,:);
TGS2600=reshapeddec(17,:)*256+reshapeddec(18,:);
TGS2602=reshapeddec(19,:)*256+reshapeddec(20,:);
TGS2620=reshapeddec(21,:)*256+reshapeddec(22,:);
MS1100=reshapeddec(23,:)*256+reshapeddec(24,:);
MR516=reshapeddec(25,:)*256+reshapeddec(26,:);
TGS2444=reshapeddec(27,:)*256+reshapeddec(28,:);
WSP2110=reshapeddec(29,:)*256+reshapeddec(30,:);
NAP55A=reshapeddec(31,:)*256+reshapeddec(32,:);
TEMP1=(reshapeddec(34,:)*256+reshapeddec(35,:))/100;
HUMD1=(reshapeddec(36,:)*256+reshapeddec(37,:))/100;
TEMP2=(reshapeddec(39,:)*256+reshapeddec(40,:))/100;
HUMD2=(reshapeddec(41,:)*256+reshapeddec(42,:))/100;

% 生成气体传感器、温湿度传感器的生成的矩阵
truevalue=[H2S',C7H8',C6H6',NO',CO',TGS822',TGS826',TGS8669',TGS2600',TGS2602',TGS2620',MS1100',MR516',TGS2444',WSP2110',NAP55A',TEMP2',HUMD2'];
dlmwrite('OriginalSignal.txt',truevalue,'delimiter', ' ');
   
originalSignal = textread('OriginalSignal.txt');          %读取文件数据
originalDATA=textread('data.dat','%s')';%读取原始数据
numofframe=length(originalDATA)/46;%46为每一帧的数据个数，numoffranme为帧数
T=(datenum(originalDATA(length(originalDATA)-44),'HH:MM:SS.FFF')-datenum(originalDATA(2),'HH:MM:SS.FFF'))*24*3600;
NDataDel=numofframe-ceil(T*8.38);

step=fix(numofframe/NDataDel);
originalSignal(1:step:step*NDataDel,:)=[];

frequency=size(originalSignal,1)/T;
t=linspace(0,T,(numofframe-NDataDel));%形成时间序列
col=size(originalSignal,2);%矩阵列数，即传感器个数

figure(1)
plot(t,originalSignal);
title('originalSignal');

%T2-T1=40 将会有336个样本
T1=str2double('115');   %通气时刻
T2=str2double('135');   %放气时刻  
t3=str2double('0');   %通气补偿时间 1.5
t4=str2double('0');   %放气响应补偿 0.15
t5=str2double('0');   %降噪补偿数时间    
t6=str2double('8.38');   %最低参考频率
lowestRowsOfA1=ceil(t6*60); %60秒采样数
lowestRowsOfA2=ceil(t6*30); %30秒采样数
N=ceil(t6*t5);      %降噪补偿数时间采样数

A1=zeros(lowestRowsOfA1,col);
A2=zeros(lowestRowsOfA2,col);
A=zeros((lowestRowsOfA1+lowestRowsOfA2+N),col);
N1=round((T1-t3)*frequency);   %通气时间－通气补偿时间
N2=N1+lowestRowsOfA1;          %通气时间＋60秒
N3=round((T2-t4)*frequency);   %放气时间－放气补偿时间
N4=N3+lowestRowsOfA2;          %放气时间＋30秒

A0=zeros(N3-N1+1,col);     %零矩阵

numbers={N,N1,N2,N3,N4};

for i = 1:col;
    A0(:,i)=originalSignal(N1:N3,i);
end
A=[A0'];
lowestRowsofN=ceil((N3-N1+1));
NewT0=linspace(0,100,lowestRowsofN);

figure(2)
plot(NewT0,A);          %绘出包括去噪补偿的图谱
title('cuttedSignal');

Signal_AfterFiltration=zeros((lowestRowsOfA1+lowestRowsOfA2),col);
dlmwrite('ValueofN.txt',numbers,'delimiter', ' ');
dlmwrite('SignalCutted.txt',A,'delimiter', ' ');

% 对数据进行标记
% 先做二分类问题，肺癌与健康做区分
listNum=size(A,2); %A矩阵的列数
AH=ones(listNum,1);
% AH=zeros(listNum,1);
B=[AH,A0];  %对数据标记，第一列数据为0为健康，为１为患有肺癌

% 划分训练集与测试集
k = round(listNum * 0.9);
C=B(randperm(listNum,k),:);
D=B(randperm(listNum,listNum-k),:);
% 
% dlmwrite('traindata.txt',C,'delimiter',' ','-append');
% dlmwrite('testdata.txt',D,'delimiter',' ','-append');


% 
% 
% %　使用RLS递推最小二乘法对原始数据进行降维操作
% 
% 
% %  使用神经网络对数据进行训练