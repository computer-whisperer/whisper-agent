# Keep kotlinx.serialization metadata for the wire types. Generated serializers
# rely on @Serializable classes retaining their companions and synthetic methods.
-keep,includedescriptorclasses class com.cjbal.whisperagent.protocol.** { *; }
-keepclasseswithmembers class ** { kotlinx.serialization.KSerializer serializer(...); }
-keepnames class kotlinx.serialization.internal.** { *; }
