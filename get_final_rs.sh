paste ./data/test_qid score.txt |awk -F"\t" '{print $4"\t"$0}' | sort -g | awk -F"\t" 'OFS="\t"{print $3, $4, $5, $6}' > final_rs.txt
