cd ./src/
python -u sogou.py
cd ..
sh ./train_and_rank.sh
sh ./get_final_rs.sh
cd ./src/
python post_process.py ../final_rs.txt ../final.txt
