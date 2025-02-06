import folium
from folium import plugins, features

def create_map(center_coordinates=(30.432555, -91.192306), zoom_start=7):
    """
    Creates a Folium map centered on given coordinates with various interactive plugins.
    
    Parameters:
        center_coordinates (tuple): Latitude and Longitude for map center.
        zoom_start (int): Initial zoom level for the map.

    Returns:
        folium.Map: A Folium map object.
    """
    # Create a Folium map object
    map_object = folium.Map(
        location=center_coordinates,
        zoom_start=zoom_start
    )

    # Add an inset mini-map to the main map
    mini_map = plugins.MiniMap(position='bottomright')
    map_object.add_child(mini_map)

    # Display cursor coordinates on the map
    coordinate_format = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"
    plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        prefix="Mouse:",
        lat_formatter=coordinate_format,
        lng_formatter=coordinate_format
    ).add_to(map_object)

    # Enable geolocation search functionality
    plugins.Geocoder(
        position='topright',
        addmarker=True
    ).add_to(map_object)

    # Enable display of latitude and longitude popups upon mouse click
    features.LatLngPopup().add_to(map_object)

    # Allow the map to toggle full screen
    plugins.Fullscreen(
        position="topright",
        title="Make fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True
    ).add_to(map_object)

    # Add drawing tool for polygons, export bounding box and geojson options
    plugins.Draw(
        export=True,
        filename="boundingbox.geojson",
        position="topleft",
        draw_options={"rectangle": {'allowIntersection': False}},
        edit_options={"poly": {'allowIntersection': False}}
    ).add_to(map_object)

    return map_object