from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from crewai_tools import BaseTool as CrewBaseTool
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from langchain_core.utils.function_calling import convert_to_openai_function
import ast
import uuid
from typing import Type, Dict
import os


class HistoricalPriceArguments(BaseModel):
    symbol: str = Field(description="The stock symbol/symbols to get the historical price for. Can be a single stock symbol or multiple stock symbols separated by a comma. Example: AAPL,MSFT")
    days: str = Field(description="The number of days to get the historical price for.")

class PlotLineChartArguments(BaseModel):
    x_values: str = Field(description="The x values represented as a string. Example format: ['2024-04-02', '2024-04-03', '2024-04-04', '2024-04-05']")
    y_values: str = Field(description="The y values represented as a string. Example format: [166, 168, 171, 164]")
    symbol: str = Field(description="The stock symbol to plot the line chart for.")

class NoneArguments(BaseModel):
    answer: str = Field(description="The answer to process manually.")

class HistoricalPriceTool:
    def _run(self, symbol: str, days: str) -> str:
        """Use the tool."""
        end_date = datetime.now()
        start_date = end_date - relativedelta(days=int(days))
        data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        close_prices = data['Close'].astype(int).to_dict()
        return str(close_prices)

class PlotLineChartTool:
    def _run(self, x_values: str, y_values: str, symbol: str) -> str:
        """Use the tool."""
        x_values = ast.literal_eval(x_values)
        y_values = ast.literal_eval(y_values)
        fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines'))
        xaxis_title = "Date"
        yaxis_title = "Price of " + symbol
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

        # Check if the directory exists and create it if necessary
        if not os.path.exists('images'):
            os.makedirs('images')

        #generate a random file name
        filename = f"images/fig_{uuid.uuid4().hex}.png"
        
        fig.write_image(filename)
        return f"Line chart has been created successfully to {filename}"

class NoneTool:
    def _run(self, answer: Dict) -> str:
        """Use the tool."""
        return str(answer)

class GetHistoricalPriceBase(HistoricalPriceTool, BaseTool):
    name: str = "historicalprice-tool"
    description: str = "Returns the price information of a stock and corresponding timestamps for the last n days."
    args_schema: Type[BaseModel] = HistoricalPriceArguments

class GetHistoricalPriceCrew(HistoricalPriceTool, CrewBaseTool):
    name: str = "historicalprice-tool"
    description: str = "Returns the price information of a stock and corresponding timestamps for the last n days."
    args_schema: Type[BaseModel] = HistoricalPriceArguments

class PlotLineChartBase(PlotLineChartTool, BaseTool):
    name: str = "line-chart-tool"
    description: str = "Creates and saves a line chart for the given stock symbol using the x and y values provided."
    args_schema: Type[BaseModel] = PlotLineChartArguments

class PlotLineChartBaseCrew(PlotLineChartTool, CrewBaseTool):
    name: str = "line-chart-tool"
    description: str = "Creates and saves a line chart for the given stock symbol using the x and y values provided."
    args_schema: Type[BaseModel] = PlotLineChartArguments

class NoneToolBase(NoneTool, BaseTool):
    name: str = "manuel-processing-tool"
    description: str = "This tool is useful when you don't have any other tool to process the answer."
    args_schema: Type[BaseModel] = NoneArguments
        
class NoneToolCrew(NoneTool, CrewBaseTool):
    name: str = "manuel-processing-tool"
    description: str = "This tool is useful when you don't have any other tool to process the answer."
    args_schema: Type[BaseModel] = NoneArguments



def get_openai_functions_definitions(tools):
    functions = [convert_to_openai_function(t) for t in tools]
    print(str(functions))
    return functions