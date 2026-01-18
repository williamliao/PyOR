#!/usr/bin/env python3
"""
台灣天氣預報 MCP 服務器
"""

from typing import Any, Dict, List, Optional
import httpx
import json
import sys
import argparse
import asyncio
from dotenv import load_dotenv
import os

# 初始化
load_dotenv()

# Constants
CWB_API_BASE = os.getenv("CWB_API_BASE")
USER_AGENT = "weather-app/1.0"
CWB_API_KEY = os.getenv("CWB_API_KEY")

# 檢查是否能導入 FastMCP
try:
    from mcp.server.fastmcp import FastMCP
    print("✅ FastMCP imported successfully", file=sys.stderr)
    
    # Initialize FastMCP server
    mcp = FastMCP("weather")
    print("✅ FastMCP server initialized", file=sys.stderr)
except ImportError as e:
    print(f"❌ Failed to import FastMCP: {e}", file=sys.stderr)
    print("Please install: pip install mcp fastmcp", file=sys.stderr)
    sys.exit(1)

async def make_cwb_request(url: str) -> dict[str, Any] | None:
    """Make a request to the CWB API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error making CWB request: {e}", file=sys.stderr)
            return None

def format_taiwan_forecast(location_data: Dict) -> str:
    """Format Taiwan forecast data into a readable string."""
    location_name = location_data.get("LocationName", "Unknown")
    weather_elements = location_data.get("WeatherElement", [])
    
    # Initialize empty dictionary to store element data
    forecast_data = {}
    
    # 建立各氣象元素的字典
    for element in weather_elements:
        element_name = element.get("ElementName", "")
        time_periods = element.get("Time", [])
        
        if element_name not in forecast_data:
            forecast_data[element_name] = []
            
        for period in time_periods:
            start_time = period.get("StartTime", "")
            end_time = period.get("EndTime", "")
            
            # 處理不同元素的值結構
            values = period.get("ElementValue", [])
            if not values:
                continue
                
            # 創建基本時間週期記錄
            period_record = {
                "startTime": start_time,
                "endTime": end_time,
            }
            
            # 根據元素類型取得對應的值
            if element_name == "天氣現象":
                period_record["parameterName"] = values[0].get("Weather", "")
                period_record["weatherCode"] = values[0].get("WeatherCode", "")
            elif element_name == "最高溫度":
                period_record["parameterName"] = values[0].get("MaxTemperature", "")
            elif element_name == "最低溫度":
                period_record["parameterName"] = values[0].get("MinTemperature", "")
            elif element_name == "12小時降雨機率":
                period_record["parameterName"] = values[0].get("ProbabilityOfPrecipitation", "")
            elif element_name == "舒適度指數" or element_name == "最大舒適度指數":
                period_record["parameterName"] = values[0].get("MaxComfortIndexDescription", "")
            else:
                # 一般情況，取第一個值
                if isinstance(values[0], dict):
                    # 找出字典中第一個非空值
                    first_key = next(iter(values[0].keys()), None)
                    if first_key:
                        period_record["parameterName"] = values[0].get(first_key, "")
                else:
                    period_record["parameterName"] = str(values[0])
            
            forecast_data[element_name].append(period_record)
    
    # 建立可讀性格式的預報字串
    forecast_str = f"🌤️ Forecast for {location_name}:\n{'='*50}\n"
    
    # 確保天氣現象存在並且有資料
    weather_periods = forecast_data.get("天氣現象", [])
    if weather_periods:
        # 依時間順序處理每個時段
        for i, period in enumerate(weather_periods):
            start_time = period["startTime"]
            end_time = period["endTime"]
            forecast_str += f"\n📅 {start_time} to {end_time}:\n"
            
            # 天氣現象
            weather = period.get("parameterName", "未知")
            forecast_str += f"   🌈 Condition: {weather}\n"
            
            # 降雨機率
            precip_periods = forecast_data.get("12小時降雨機率", [])
            if precip_periods and i < len(precip_periods):
                pop = precip_periods[i]["parameterName"]
                if pop != "-":
                    forecast_str += f"   🌧️  Precipitation Chance: {pop}%\n"
            
            # 溫度
            min_temp_periods = forecast_data.get("最低溫度", [])
            max_temp_periods = forecast_data.get("最高溫度", [])
            
            if min_temp_periods and max_temp_periods and i < len(min_temp_periods) and i < len(max_temp_periods):
                min_temp = min_temp_periods[i]["parameterName"]
                max_temp = max_temp_periods[i]["parameterName"]
                forecast_str += f"   🌡️  Temperature: {min_temp}-{max_temp}°C\n"
            
            # 舒適度
            comfort_periods = forecast_data.get("最大舒適度指數", [])
            if not comfort_periods:
                comfort_periods = forecast_data.get("舒適度指數", [])
                
            if comfort_periods and i < len(comfort_periods):
                comfort = comfort_periods[i]["parameterName"]
                forecast_str += f"   😊 Comfort Index: {comfort}\n"
                
            # 風向風速
            wind_dir_periods = forecast_data.get("風向", [])
            wind_speed_periods = forecast_data.get("風速", [])
            
            if wind_dir_periods and wind_speed_periods and i < len(wind_dir_periods) and i < len(wind_speed_periods):
                wind_dir = wind_dir_periods[i]["parameterName"]
                wind_speed = wind_speed_periods[i]["parameterName"].split(',')[0] if ',' in wind_speed_periods[i]["parameterName"] else wind_speed_periods[i]["parameterName"]
                forecast_str += f"   💨 Wind: {wind_dir} {wind_speed} m/s\n"
    else:
        forecast_str += "\n❌ No detailed forecast available for this location.\n"
    
    return forecast_str

@mcp.tool()
async def get_taiwan_forecast(location: str = "", limit: int = 23, sort: str = "time") -> str:
    """Get weather forecast for Taiwan locations from Central Weather Bureau.
    
    Use this tool when the user asks about:
    - Taiwan weather forecast (台灣天氣預報)
    - Weather in Taiwan cities (台北、高雄、台中等城市天氣)
    - Temperature, rain chance, or weather conditions in Taiwan

    TRIGGER KEYWORDS: weather, forecast, 天氣, 氣象, temperature, rain, 下雨
    
    Args:
        location: Taiwan city/county name in Traditional Chinese
                 Examples: 臺北市, 新北市, 高雄市, 台中市, 台南市
                 Leave empty to get forecasts for multiple locations
        limit: Maximum number of results (default: 23, max: 100)
        sort: Sort by 'time' (default) or other fields
    
    Returns:
        Formatted weather forecast with:
        - Time period
        - Weather condition
        - Temperature range
        - Precipitation chance
        - Wind information
        - Comfort index
    
    Examples:
        - get_taiwan_forecast(location="臺北市") -> Taipei weather
        - get_taiwan_forecast(location="高雄市") -> Kaohsiung weather
        - get_taiwan_forecast() -> Multiple locations
    """
    print(f"查詢台灣天氣預報: 地區={location}, 限制={limit}, 排序={sort}", file=sys.stderr)
    
    # 嘗試從 API 獲取資料
    url = f"{CWB_API_BASE}/v1/rest/datastore/F-D0047-091?Authorization={CWB_API_KEY}&limit={limit}&format=JSON&sort={sort}"
    
    # 如果指定了地點，加上 locationName 參數
    if location:
        url += f"&locationName={location}"
    
    data = await make_cwb_request(url)
    
    # 如果 API 失敗，嘗試從本地文件讀取
    if not data:
        try:
            with open("response_1743144846646.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return "❌ 無法連線到氣象局 API 且無法讀取本地備份資料"
    
    # 處理資料
    try:    
        if not data or "records" not in data:
            return "❌ No forecast data available for Taiwan."
        
        # 檢查數據結構
        locations = None
        if "Locations" in data["records"]:
            # 新的API結構
            if data["records"]["Locations"] and isinstance(data["records"]["Locations"], list):
                locations = data["records"]["Locations"][0].get("Location", [])
        elif "location" in data["records"]:
            # 舊的API結構
            locations = data["records"]["location"]
        
        if not locations:
            return "❌ 無法解析氣象資料的位置資訊。"
        
        if not location:
            # Return forecasts for all locations (limited)
            forecasts = []
            for loc_data in locations[:3]:  # Limit to first 3 locations
                forecasts.append(format_taiwan_forecast(loc_data))
            return "\n\n" + "="*80 + "\n\n".join(forecasts)
        else:
            # Find the specific location
            for loc_data in locations:
                if loc_data.get("LocationName") == location:
                    return format_taiwan_forecast(loc_data)
            
            # If location not found
            available_locations = [loc.get("LocationName", "") for loc in locations[:10]]
            return f"❌ Location '{location}' not found.\n\n🏙️ Available locations: {', '.join(available_locations)}"
            
    except Exception as e:
        return f"❌ 處理天氣資料時出錯: {str(e)}"

async def test_function():
    """測試函數"""
    print("🧪 Testing weather forecast tool...")
    result = await get_taiwan_forecast("新北市")
    print("結果:")
    print(result)
    return result

async def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Get Taiwan weather forecast")
    parser.add_argument("--location", default="", help="Location name in Taiwan")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of results")
    parser.add_argument("--sort", default="time", help="Sort field")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--check", action="store_true", help="Check if server is ready")
    
    args = parser.parse_args()
    
    if args.check:
        print("✅ Python MCP server is ready!", file=sys.stderr)
        print(f"✅ FastMCP imported: {mcp is not None}", file=sys.stderr)
        print(f"✅ Environment loaded: CWB_API_BASE={CWB_API_BASE}", file=sys.stderr)
        return
    
    if args.test:
        print("🧪 Testing weather forecast tool...")
        result = await test_function()
    else:
        result = await get_taiwan_forecast(args.location, args.limit, args.sort)
        print(result)
    
    return result

if __name__ == "__main__":
    # 檢查命令行參數
    if len(sys.argv) > 1:
        if "--test" in sys.argv or "--check" in sys.argv or "--location" in sys.argv:
            # 命令行模式
            print("📋 Running in command line mode...", file=sys.stderr)
            asyncio.run(main())
            sys.exit(0)
    
    # 以 MCP 模式運行
    print("🌤️ Starting Weather MCP Server...", file=sys.stderr)
    print("🔌 Ready to accept MCP connections...", file=sys.stderr)
    
    try:
        # 確保使用 stdio 傳輸
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("👋 Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)