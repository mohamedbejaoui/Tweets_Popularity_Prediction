### Tweets popularity preduction

#### Needed libraries
<ul>
  <li>[cppkafka](https://github.com/mfontanini/cppkafka)</li>
  <li>[nlohmann json](https://github.com/nlohmann/json)</li>
</ul>

#### Using
In order to build, just run in the main directory:
> mkdir build<br/>
> cd build<br/>
> cmake ..<br/>
> make<br/>

Then, run a kafka server by typing:
> bin/zookeeper-server-start.sh config/zookeeper.properties<br/>
> bin/kafka-server-start.sh config/server.properties<br/>
These two commands should be run inside the kafka folder where it was installed

Finally, inside the build folder created previously run the consumer followed by the producer:
> ./testconsumer<br/>
> ./testproducer<br/>
