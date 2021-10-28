
line="------------------------------" #在定义bash variable的时候，‘=’前后不可以留空格

echo "starting at : $(date)" ; echo $line 

echo "UPTIME" ; UPTIME ; echo $line 

echo "FREE" ; FREE ;echo $line 

echo "WHO" ; WHO ; echo $line 

echo "finishing at : $(date)" # $() ensures that the result of the command inside of it will be passed to the outside command