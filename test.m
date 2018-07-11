%测试训练好的模型
[Eclass,EH2S,EC7H8,EC6H6,ENO,ECO,ETGS822,ETGS826,ETGS8669,ETGS2600,ETGS2602,ETGS2620,EMS1100,EMR516,ETGS2444,EWSP2110,ENAP55A,ETEMP2,EHUMD2]=textread('testdata.txt');
TESTMEXTRIA=[EH2S,EC7H8,EC6H6,ENO,ECO,ETGS822,ETGS826,ETGS8669,ETGS2600,ETGS2602,ETGS2620,EMS1100,EMR516,ETGS2444,EWSP2110,ENAP55A,ETEMP2,EHUMD2];

% listB=size(MEXTRIB,1);
% K = ceil(0.3*listB);
% MEXTRIA=MEXTRIB(randperm(listB,K),:);

%取出矩阵mextria中的特征值与类别
% FEATURE=MEXTRIB(:,(2:19));
% CLASS=MEXTRIB(:,1);
% 
[testInput,PS] = mapminmax(TESTMEXTRIA');
testOutput = sim(net, testInput);


% [testInput] = tramnmx(TESTMEXTRIA');
% testOutput = sim(net, testInput, minI, maxI);

%统计测试集正确率
[s1,s2] = size(testOutput);
hitNum = 0;
for i = 1: s2
    [m, index] = max(testOutput(:,i));
    if(index == Eclass(i)+1)
        hitNum = hitNum +1;
    end
end
correctRate = 100*hitNum/s2
sprintf('识别率为%3.3f%%',correctRate);