CREATE TABLE "public".config (
	dataset_filename varchar NULL,
	satellite_grid varchar NULL,
	satellite_config json NULL,
	id serial NOT NULL,
	"indicator" varchar NULL,
	"comment" varchar NULL,
	"output" varchar NOT NULL DEFAULT 'regression'::character varying,
	log bool NULL DEFAULT false,
	base_raster_aggregation int2 NULL,
	nightlights_date json NULL,
	"scope" varchar NULL,
	"NDs_date" json NULL,
	PRIMARY KEY (id)
)
