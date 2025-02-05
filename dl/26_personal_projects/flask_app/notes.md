# RESTful API

- An application programming interface (API) defines the rules that you must follow to communicate with other software systems. 
- RESTful API is an interface that two computer systems use to exchange information securely over the internet. 
  - They follow secure, reliable, and efficient software communication standards
- Clients are users who want to access information from the web.
-  Organizations use APIs to share resources and provide web services while maintaining security, control, and authentication.
- REST
  - Representational State Transfer (REST) is a software architecture that imposes conditions on how an API should work. 
  - Uniform interface: The server transfers information in a standard format, e.g. json
    - This can be different from how the server internally represents the resource. 
    - Serializing is the process of converting an object's data into a format that can be easily stored or transmitted, e.g. json. 
    - Deserializing is the process of converting stored data back into a usable object in memory
  - Statelessness: Server completes every client request independently of all previous requests
  - Layered system
  - Cacheability
  - Code on demand
- Benefits
  - Scalability
  - Flexibility
  - Independence
- Typical process
  - The client sends a request to the server. The client follows the API documentation to format the request in a way that the server understands.
  - The server authenticates the client and confirms that the client has the right to make that request.
  - The server receives the request and processes it internally.
  - The server returns a response to the client. The response contains information that tells the client whether the request was successful. The response also includes any information that the client requested.
- Requests typically contain the following main components
  - Unique resource identifier
    - For REST services, the server typically performs resource identification by using a Uniform Resource Locator (URL). The URL specifies the path to the resource and is also called the request endpoint.
  - Method
    - Developers often implement RESTful APIs by using the Hypertext Transfer Protocol (HTTP). 
    - 4 common HTTP methods are GET, POST, PUT, DELETE. 
  - HTTP headers
    - Metadata e.g. format of the request and response, provides information about request status, auth details
- Authentication methods
  - HTTP Authentication
    - Basic: username and password
    - Bearer: Token
  - API Keys
  - OAuth 
    - Password + Token
- Responses typically contain the following main components
  - Status line
  - Message body: The server selects an appropriate representation format based on what the request headers contain. 
  - Headers: context, information like server, encoding, date, and content type
- Implementation
  - Current app is built with the tutorial from [Krebs and Martinez](https://auth0.com/blog/developing-restful-apis-with-python-and-flask/), which covers basic concepts
  - We can easily extend this to LLM-specific use cases like [this tutorial](https://medium.com/@shmilysyg/setup-rest-api-service-of-ai-by-using-local-llms-with-ollama-eb4b62c13b71) (Ollama + FastAPI)
