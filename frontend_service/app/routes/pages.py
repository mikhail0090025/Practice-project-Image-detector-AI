from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import aiohttp
from app.dependencies import get_http_client

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/results", response_class=HTMLResponse)
async def get_results(request: Request, client: aiohttp.ClientSession = Depends(get_http_client)):
    try:
        async with client.get("http://neural_service:8001/predict") as response:
            if response.status == 200:
                data = await response.json()
            else:
                data = {"error": "Не удалось получить данные от нейронной сети"}
    except:
        data = {"error": "Ошибка подключения к нейронной сети"}
    return templates.TemplateResponse("results.html", {"request": request, "data": data})

@router.get("/charts", response_class=HTMLResponse)
async def get_charts(request: Request, client: aiohttp.ClientSession = Depends(get_http_client)):
    try:
        async with client.get("http://chart_service:8002/chart_data") as response:
            if response.status == 200:
                chart_data = await response.json()
            else:
                chart_data = {"error": "Не удалось получить данные для графиков"}
    except:
        chart_data = {"error": "Ошибка подключения к сервису графиков"}
    return templates.TemplateResponse("charts.html", {"request": request, "chart_data": chart_data})