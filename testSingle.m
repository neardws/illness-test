singleSample=[0.100000000000000,8,10.6000000000000,0.150000000000000,2,2419,1619,2934,3763,1900,3447,4832,806,2831,1867,58,38.4600000000000,52.0400000000000];
% singleSample=B(:,(2:19));
[singleInput]=tramnmx(singleSample',minI, maxI);
singleOutput = sim(net,singleInput)