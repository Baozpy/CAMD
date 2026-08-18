cat ./bug_list.txt | while read bug
do
    mvn exec:java -Dexec.mainClass="Main" -Dexec.args="$bug"
done
