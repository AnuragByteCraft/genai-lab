import os
from typing import Optional
import requests
from mcp.server.fastmcp import FastMCP

# Replace with your OpenWeatherMap API key
# API_KEY = os.getenv('default', 'e3b9c6a368fedd2be5915430643f4704')
API_KEY = "357e8af792550a8f93ef93d0234261a1"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str, units: str = "metric") -> Optional[str]:
    """
    Get current weather information for a location.
    
    Args:
        location: The city name or zip code to get weather for
        units: 'metric' or 'imperial' for temperature units (default: metric)
        
    Returns:
        Formatted weather information string or None if error occurs
    """
    try:
        params = {
            'q': location,
            'appid': API_KEY,
            'units': units
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            
            return (f"Weather in {location}:\n"
                   f"Condition: {weather}\n"
                   f"Temperature: {temp}°{'C' if units == 'metric' else 'F'}\n"
                   f"Humidity: {humidity}%\n"
                   f"Wind Speed: {wind_speed} m/s")
        else:
            return f"Error getting weather data: {data.get('message', 'Unknown error')}"
            
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="streamable-http")


