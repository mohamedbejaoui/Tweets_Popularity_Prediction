gnome-terminal --geometry 400x60 --tab --title="zookeeper" -e 'bash -c "cd ~/kafka/ && bin/zookeeper-server-start.sh config/zookeeper.properties"' \
--tab --title="kafka" -e 'bash -c "sleep 3 && cd ~/kafka/ && bin/kafka-server-start.sh config/server.properties"' \
--tab --title="config" -e 'bash -c "sleep 15 && source ~/anaconda3/etc/profile.d/conda.sh && conda activate <put_conda_env_name_here> && python kafka_config.py"' \
--tab --title="tweet_process"  -e 'bash -c "sleep 20 && source ~/anaconda3/etc/profile.d/conda.sh && conda activate <put_conda_env_name_here> && python hawkes_params_fitter.py"' \
--tab --title="alert_receiver" -e 'bash -c "sleep 18 && source ~/anaconda3/etc/profile.d/conda.sh && conda activate <put_conda_env_name_here> && python cascade_size_alert_receiver.py"' \
--tab --title="tweet_reader" -e 'bash -c "sleep 20 && source ~/anaconda3/etc/profile.d/conda.sh && conda activate <put_conda_env_name_here> && python tweets_data_reader.py"' \
--tab --title="rf_trainer" -e 'bash -c "sleep 18 && source ~/anaconda3/etc/profile.d/conda.sh && conda activate <put_conda_env_name_here> && python random_forest_trainer.py"'
