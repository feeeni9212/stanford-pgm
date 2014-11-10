% This script builds modifies the TestI0 to include different test
% characteristics and make the decision node depend on the test variable.

load TestI0.mat;
D = TestI0.DecisionFactors(1);
F = TestI0.RandomFactors(10);
[meu0, optD0] = OptimizeWithJointUtility(TestI0);
fprintf(1, 'No information:\nMax Expected Utility = %d\n\n', meu0);

% test option 1
D.var = [D.var 11];
D.card = [D.card 2];
TestI0.DecisionFactors = [D];
[meu1, optD1] = OptimizeWithJointUtility(TestI0);
fprintf(1, 'Test option 1:\nMax Expected Utility = %d\n', meu1);
vpi1 = meu1-meu0;
fprintf(1, 'VPI = %d\n', vpi1);
fprintf(1, 'Monetary Worth = %d\n', exp(vpi1/100)-1);
fprintf(1, '\n');

% test option 2
F.val = F.val(end:-1:1);
TestI0.RandomFactors(10) = F;
[meu2, optD2] = OptimizeWithJointUtility(TestI0);
fprintf(1, 'Test option 2:\nMax Expected Utility = %d\n', meu2);
vpi2 = meu2-meu0;
fprintf(1, 'VPI = %d\n', vpi2);
fprintf(1, 'Monetary Worth = %d\n', exp(vpi2/100)-1);
fprintf(1, '\n');

% test option 3
F.val = [0.999 0.001 0.001 0.999];
TestI0.RandomFactors(10) = F;
[meu3, optD3] = OptimizeWithJointUtility(TestI0);
fprintf(1, 'Test option 2:\nMax Expected Utility = %d\n', meu3);
vpi3 = meu3-meu0;
fprintf(1, 'VPI = %d\n', vpi3);
fprintf(1, 'Monetary Worth = %d\n', exp(vpi3/100)-1);
fprintf(1, '\n');
