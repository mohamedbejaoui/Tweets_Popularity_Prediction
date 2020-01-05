#include <iostream>
#include <cppkafka/cppkafka.h>

using namespace cppkafka;

bool running = true;

int main() {

  std::string topic_name = "tweets_stream";

  // Construct the Configuration
  Configuration config = {
    {"metadata.broker.list", "127.0.0.1:9092"},
    {"group.id", "unk_group"},
    {"enable.auto.commit", false}
  };

  // Create the consumer
  Consumer consumer(config);

  // Subscribe to the topic
  consumer.subscribe({ topic_name });

  std::cout << "Consuming messages from topic" << topic_name << std::endl;

  // Now read lines and write them into kafka
  while (running) {
    Message msg = consumer.poll();
    if (msg) {
      if (msg.get_error()) {
        // Ignore EOF notifications from rdkafka
        if (!msg.is_eof()) {
            std::cout << "[+] Received error notification: " << msg.get_error() << std::endl;
        }
      }
      else {
        // Print the key (if any)
        if (msg.get_key()) {
          std::cout << msg.get_key() << " -> ";
        }
        // Print the payload
        std::cout << msg.get_payload() << std::endl;
        // Now commit the message
        consumer.commit(msg);
      }
    }
  }

  return 0;
}
