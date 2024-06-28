import folium
# Create a map centered at a specific location
lat_long={'Cambedoo': [-32.22039,24.5128,'red'],
'Mountain Zebra':[-32.14095,25.50956,'orange'],
'Serengeti':[-2.3333,34.83333,'black'],
'Karoo':[-32.36344,22.5412,'darkgreen'],
'Kruger':[-23.98838,31.55474,'blue'],
'Enonkishu':[-1.07642,35.24681,'purple'],
'Kgalagadi':[-25.25221,20.9717, 'pink']}

map_center = (-9, 26.75)
my_map = folium.Map(location=map_center, zoom_start=12)

# Add pins to the map
for name in lat_long:
    coordinates = lat_long[name][:-1]
    color = lat_long[name][-1]
    folium.Marker(location=coordinates, popup=name, icon=folium.Icon(color=color)).add_to(my_map)

my_map.save("map_with_pins.html")

# Convert the HTML file to a PNG image using an external tool like wkhtmltoimage
# Ensure wkhtmltoimage is installed on your system: https://wkhtmltopdf.org/
# import subprocess
# subprocess.run(["wkhtmltoimage", "map_with_pins.html", "map_with_pins.png"])