from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import aiohttp
from app.dependencies import get_http_client
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/results", response_class=HTMLResponse)
async def get_results(request: Request, client: aiohttp.ClientSession = Depends(get_http_client)):
    try:
        async with client.get("http://neural_service:5001/predict") as response:
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
        async with client.get("http://chart_service:5002/chart_data") as response:
            if response.status == 200:
                chart_data = await response.json()
            else:
                chart_data = {"error": "Не удалось получить данные для графиков"}
    except:
        chart_data = {"error": "Ошибка подключения к сервису графиков"}
    return templates.TemplateResponse("charts.html", {"request": request, "chart_data": chart_data})

@router.get("/about", response_class=HTMLResponse)
async def get_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@router.get("/profile_1", response_class=HTMLResponse)
async def get_profile_1(request: Request):
    return templates.TemplateResponse("profile_1.html", {"request": request})

@router.get("/profile_2", response_class=HTMLResponse)
async def get_profile_2(request: Request):
    return templates.TemplateResponse("profile_2.html", {"request": request})

@router.get("/profile_3", response_class=HTMLResponse)
async def get_profile_3(request: Request):
    return templates.TemplateResponse("profile_3.html", {"request": request})

@router.get("/profile_4", response_class=HTMLResponse)
async def get_profile_4(request: Request):
    return templates.TemplateResponse("profile_4.html", {"request": request})

@router.get("/statistic", response_class=HTMLResponse)
async def get_statistic(request: Request):
    return templates.TemplateResponse("statistic_page.html", {"request": request})

@router.get("/teach", response_class=HTMLResponse)
async def get_teach(request: Request):
    return templates.TemplateResponse("teach.html", {"request": request})

@router.get("/test", response_class=HTMLResponse)
async def get_test(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

@router.get("/image/T_Vision_logo-GREEN.png", response_class=FileResponse)
async def get_image_green():
    image_path = os.path.join("frontend_service/app/static", "T_Vision_logo-GREEN.png")
    return FileResponse(image_path, media_type="image/png")

@router.get("/image/T_Vision_logo-GREEN-BLACK.png", response_class=FileResponse)
async def get_image_green_black():
    image_path = os.path.join("frontend_service/app/static", "T_Vision_logo-GREEN-BLACK.png")
    return FileResponse(image_path, media_type="image/png")

@router.get("/image/True_Vision_logo-text-GREEN.png", response_class=FileResponse)
async def get_image_text_green():
    image_path = os.path.join("frontend_service/app/static", "True_Vision_logo-text-GREEN.png")
    return FileResponse(image_path, media_type="image/png")