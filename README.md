# Qdrant Fantasy Sports

This repository contains a FastAPI application for providing fantasy sports suggestions using Qdrant as a vector database. The application takes user preferences and returns relevant player suggestions.

## Features

- Get player recommendations based on sport, team, and player preferences.
- FastAPI framework for building APIs.
- Qdrant client for handling vector data.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/manas95826/Qdrant-fantasy-sports.git
   cd Qdrant-fantasy-sports
   ```

2. Install the required packages:
   ```bash
   pip install fastapi uvicorn qdrant-client
   ```

3. Set up your Qdrant environment. Make sure to have Qdrant running locally or in a Docker container. If using Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. Optionally, set the `QDRANT_URL` environment variable if your Qdrant instance is hosted elsewhere.

## Usage

1. Run the FastAPI application:
   ```bash
   uvicorn app:app --reload
   ```

2. Open your browser and go to `http://localhost:8000/docs` to access the API documentation.

3. Use the `/suggestions/` endpoint to get player suggestions based on user preferences.

## Example Request

```json
{
  "sport": "Basketball",
  "team": "Lakers",
  "player": "LeBron James"
}
```

## Example Response

```json
{
  "suggestions": [
    {
      "player": "Player A",
      "team": "Team X",
      "sport": "Basketball",
      "score": 95
    },
    {
      "player": "Player B",
      "team": "Team Y",
      "sport": "Basketball",
      "score": 88
    }
  ]
}
```

## Author

- **Manas Chopra** - [manas95826](https://github.com/manas95826)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

You can create a `README.md` file in your repository and paste the above content into it. Adjust any details as necessary to better fit your project or personal preferences!
