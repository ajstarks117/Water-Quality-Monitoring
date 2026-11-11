import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export default function StationMap({ stations, selected, onSelect }) {
  const host = useRef(null);
  const resetExtent = useRef(null);
  const activeMap = useRef(null);
  const markers = useRef(new Map());
  const [tileError, setTileError] = useState(false);

  useEffect(() => {
    const map = L.map(host.current, { scrollWheelZoom: false });
    activeMap.current = map;
    const token = import.meta.env.VITE_MAPBOX_TOKEN;
    const tiles = token
      ? L.tileLayer(
          "https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/512/{z}/{x}/{y}?access_token=" +
            encodeURIComponent(token),
          {
            tileSize: 512,
            zoomOffset: -1,
            minZoom: 1,
            maxZoom: 18,
            attribution:
              '<a href="https://www.mapbox.com/">© Mapbox</a> · <a href="https://www.openstreetmap.org/copyright">© OpenStreetMap</a> · <a href="https://apps.mapbox.com/feedback/">Improve this map</a>',
          },
        )
      : L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 18,
          attribution:
            '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        });
    setTileError(false);
    tiles.on("tileerror", () => setTileError(true)).addTo(map);
    const bounds = L.latLngBounds([]);
    for (const s of stations) {
      const { Latitude: lat, Longitude: lon } = s;
      if (
        !Number.isFinite(lat) ||
        !Number.isFinite(lon) ||
        Math.abs(lat) > 90 ||
        Math.abs(lon) > 180
      )
        continue;
      const point = [lat, lon];
      bounds.extend(point);
      const label = document.createElement("span");
      label.textContent = s.Station + " · " + s.State;
      const marker = L.circleMarker(point, {
        radius: 5,
        weight: 1,
        color: "#fff",
        fillColor: "#087c78",
        fillOpacity: 0.85,
      })
        .addTo(map)
        .bindPopup(label)
        .on("click", () => onSelect(s));
      markers.current.set(s, marker);
    }
    resetExtent.current = () =>
      bounds.isValid()
        ? map.fitBounds(bounds, { padding: [22, 22], maxZoom: 12 })
        : map.setView([22, 80], 4);
    resetExtent.current();
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      markers.current.clear();
      map.remove();
    };
  }, [stations, onSelect]);

  useEffect(() => {
    const marker = markers.current.get(selected);
    if (marker) marker.openPopup();
    else activeMap.current?.closePopup();
  }, [selected, stations]);

  return (
    <>
      <div
        className="station-map"
        ref={host}
        aria-label="Geographic station map"
      />
      <button
        className="map-reset"
        type="button"
        onClick={() => resetExtent.current?.()}
      >
        Fit stations
      </button>
      {selected &&
        !markers.current.has(selected) &&
        (!Number.isFinite(selected.Latitude) ||
          !Number.isFinite(selected.Longitude)) && (
          <small>No coordinates available for this station.</small>
        )}
      {tileError && (
        <small role="status">
          Basemap tiles unavailable or incomplete. Markers show recorded
          coordinates.
        </small>
      )}
    </>
  );
}
