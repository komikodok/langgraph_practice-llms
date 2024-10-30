from app import app
from PIL import Image
import io

png = app.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(png))
image.show()