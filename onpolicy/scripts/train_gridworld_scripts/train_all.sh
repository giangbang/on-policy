# #!/bin/sh

# for f in *.sh; do
#     echo "$f" 
#     if [ "$f" = "train_all.sh" ]; then
#         echo ""
#     else
#         bash "$f" 
#     fi;
# done

./train_plan4_ns_mgda.sh
./train_plan4_ns_mgdapp.sh
./train_plan4_ns_mappo.sh