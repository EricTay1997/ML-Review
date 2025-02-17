# Deep Dives

## Concepts

- APIs
  - REST
    - HTTP verbs (GET/PUT/POST/PATCH/DELETE) are performed wherever possible. Note that HTTP is just a common implementation of REST.
    - Pros: Structured way of getting information from DB
    - Cons: Requires requests for each type of entity in DB. Less space efficient as RPC.
  - Remote Procedure Call (RPC, family style)
    - Allows the execution of a procedure or command in a remote machine. You can write code that executes on another computer internally the same way you write code that runs on a current machine. 
    - Pros: More space efficient, makes development easier.
    - Cons: Less uniform - not suitable for broad client compatibility.
  - GraphQL
    - You structure the data in graphs. It enables building a request to fetch exactly the data you need in one call.
    - Pros: Works well for customer-facing web and mobile applications. Frontend devs can craft their own requests to get and modify information without needing backend devs to build more routes.
    - Cons: Upfront work, less friendly for external users, and not suitable when data needs to be aggregated on the backend.
- Stateless architecture: 
  - State data (e.g. session data) is stored in a shared data store (message broker or database) and kept out of web servers. 
  - This allows for more horizontal scaling while maintaining stateful communication, but may hinder performance in scenarios requiring frequent state retrieval.
- Scaling
  - Vertical
    - Adding more power to the existing machine. This is used when its more efficient to keep certain processes together, e.g. due to geographical restrictions.
  - Horizontal
    - Add more servers
    - Shard databases
      - Vertical: By feature
      - Key-based
      - Directory-based: Need to maintain a lookup table
    - Normalization
      - Reduces redundancy but slows down joins
- Database replication
  - Usually master/slave relationship
  - A master generally only supports writes and a slave gets copies from the master and only supports reads.
  - One may have more masters for reliability, but trade off some consistency.
- Databases (SQL vs NoSQL)
  - SQL: relational database that is composed of tables where each row reflects a data entity and each column defines specific information about the field.
    - Pros: Querying, ACID (Atomicity, Consistency, Isolation, Durability)
    - Cons: Slower to write to (Use of B-Trees necessitates rewriting). Schema needs to be known ahead of time / Does not work well for mixed schema data.
  - NoSQL: Nested Key-Value store, document databases, columnar databases (when search for value), graph databases. 
    - Pros: Faster writes. Managed NoSQL services come with sharding and scaling out the box. Flexible for when schema changes rapidly e.g. startup.
    - Cons: Limited in the types of efficient queries that can be done. Less suitable where strong consistency is required.
  - ACID properties
    - Atomicity: Transactions are either all or nothing, if a transaction fails, the entire transaction is rolled back.
    - Consistency: Transactions bring the database from one valid state to another, adhering to all predefined rules and constraints. I.e. any transaction that breaks such constraints would fail and be rolled back.
    - Isolation: Transactions run independently, and concurrent transactions occur as though they were occurring sequentially. 
    - Durability: Committed transactions are permanently recorded in the database, even after system failure
  - CAP Theorem
    - A distributed data store can only provide two of the following three guarantees
      - Consistency: Every read receives the most recent write or an error. Alternatively, all nodes will give the same response to a given request.
      - Availability: Every request receives a non-error response, even when nodes are down or unavailable
      - Partition tolerance: The system continues to operate even if there is a communication break between two nodes
    - Note: The consistency here is _different_ from the consistency in ACID!
- MapReduce
  - Specifies logic to hash requests and therefore parallelize them.
- Consistent hashing
  - Hash our nodes on a ring - allows for easy reallocation. 
  - Virtual nodes to "even out" load
- Communication protocols for real-time client updates:
  - ![client_updates.png](client_updates.png)[Source](https://www.hellointerview.com/learn/system-design/deep-dives/realtime-updates)
  - Simple polling: Client makes request at regular intervals
  - Long polling: Server holds request open until it has new data. Client makes request again the moment it receives a response. 
  - Server Sent Events (SSE): Allows the server to push updates to the client, via a single, long-lived HTTP connection. Not supporting client-to-server message makes SSE simpler to implement and integrate into existing HTTP infrastructure, such as load balancers and firewalls, without the need for special handling.
  - Websockets: realtime, bidrectional communication. A common pattern is to use a message broker to handle communication between the client and server, and for backend services to communicate with this message broker.
  - WebRTC: enables direct peer-to-peer communication between browsers, perfect for video/audio calls and some data sharing like document editors
- Security
  - Authentication / Authorization: API Gateway
  - Encryption: Protocol encryption (data in transit) and storage encryption (data at rest)
    - HTTPS is the SSL/TLS protocol that encrypts data in transit
    - Use a database that supports encryption or encrypt it yourself before storing
  - Data Protection
    - Rate limiting or request throttling
- Monitoring
  - Infrastructure: CPU usage, memory usage, disk usage, network usage: Datadog or New Relic
  - Service-level: Request latency, error rates, throughput
  - Application-level: Number of users, number of active sessions, number of active connections: Google Analytics or Mixpanel
- Database Indexing
  - ![indexing.png](indexing.png)[Source](https://www.hellointerview.com/learn/system-design/deep-dives/db-indexing)
  - B-Tree Indexes
    - Balanced tree that maintains sorted data
  - Hash Indexes
    - Speed
    - Don't need range queries
    - Have more memory
  - Geospatial Indexes
    - Geohash (hash a grid)
    - Quadtree (recursively subdivide grid)
    - R-Tree (flexible, overlapping rectangles)
  - Inverted Indexes

## Additional Examples
- Servers: AWS, Azure, GCP
- Service: Kubernetes can auto-scale the number of pods
- Data Processing / Feature Pipelines
  - Batch computation engines: Apache Spark, MapReduce
  - Streaming computation engines: Apache Flink, KSQL, Spark Streaming.
- Distributing Model Training: Tensorflow's YARN
- Deployment service: AWS SageMaker, GCP Vertex AI, Azure Azure ML, Alibaba Machine Learning Studio
- Model Repos: S3
- Schedulers: Slurm, Google Borg
- Orchestrators: K8, HashiCorp Nomad, Airflow 
- Workflow management tools: Airflow, Argo, Prefect, Kubeflow, MLFlow, TFX
- Encoders:
  - Words - GLoVe
  - Text - DistilmBERT
  - Image - CLIP, SimCLR
  - Video - VideoMoCo

## Deep Dives

### K8s

### Kafka

### Redis

### PostgreSQL

### Cassandra

### DynamoDB

### Elasticsearch
- Change Data Capture
