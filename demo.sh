gnome-terminal --geometry 400x60 --tab --title="zookeeper" -e 'bash -c "cd ~/kafka/ && bin/zookeeper-server-start.sh config/zookeeper.properties"' \
--tab --title="kafka" -e 'bash -c "sleep 3 && cd ~/kafka/ && bin/kafka-server-start.sh config/server.properties"' \
--tab --title="rf_node" -e 'bash -c "sleep 10 && cd ~/Documents/PFE-SD9/protoprojetsd9/ && source ~/anaconda3/bin/activate && python kafka-config.py && python random_forest_trainer.py"' \
--tab --title="hawkes_node"  -e 'bash -c "sleep 16 && cd ~/Documents/PFE-SD9/protoprojetsd9/ && source ~/anaconda3/bin/activate && python hawkes_params_fitter.py"' \
--tab --title="size_receiver" -e 'bash -c "sleep 20 && cd ~/Documents/PFE-SD9/protoprojetsd9/ && source ~/anaconda3/bin/activate && python python cascade_size_alert_receiver.py"' \
--tab --title="tweet_reader" -e 'bash -c "sleep 24 && cd ~/Documents/PFE-SD9/protoprojetsd9/ && source ~/anaconda3/bin/activate && python python tweets_data_reader.py"'